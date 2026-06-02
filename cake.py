from llm_client import OPENAIClient
import numpy as np
import re
import math
import torch
from torch.optim import Adam
from gpytorch.kernels import RBFKernel, PeriodicKernel, LinearKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll


SYSTEM_PROMPT_TEMPLATE = """
You are an expert in machine learning, specializing in Gaussian processes. Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can be captured by a kernel function. 
You can use any of the following base kernels: {base_kernels}, and combine these kernels using the following operators: {operators}. 
Your goal is to construct a kernel expression that best explains the observed data. 
The kernel will be evaluated using a fitness score normalized between [0, 1], where higher values indicate better fit to the data.
"""

CROSSOVER_PROMPT_TEMPLATE = """
You are given two parent kernels and their fitness scores:  
{parent_kernel1} ({fitness1}),  {parent_kernel2} ({fitness2}) 

Please propose a new kernel that has a potentially higher fitness score. 
You may combine the parent kernels using any of the operators from: {operators}. 
Briefly explain your reasoning behind the proposed kernel.
"""

MUTATION_PROMPT_TEMPLATE = """
You are given a kernel and its fitness score:  
{kernel} ({fitness})

Please propose a new kernel that has a potentially higher fitness score. 
You may replace a base kernel in the current expression with another base kernel from the set: {base_kernels}. 
Briefly explain your reasoning behind the proposed kernel.
"""

class CAKE:

    def __init__(self, n_c = 1, n_p = 4, p_m = 0.7, model_name = "gpt-4o-mini", 
                device = "cpu"):
        self.n_c = n_c
        self.n_p = n_p
        self.p_m = p_m
        self.operators = ["+", "*"]
        self.base_kernels = ["SE", "PER", "LIN", "RQ", "M3", "M5"]
        self.K = {kernel: {} for kernel in self.base_kernels}
        self.model_name = model_name
        self.device = device
        self.llm = OPENAIClient(model_name= self.model_name)
        pass


    
    def evaluate_fitness(self):
        for k in self.K:
            model, likelihood, bic = self.gp_surrogate(self.train_x, self.train_y, kernel = k, 
                                                       device = self.device)
            self.K[k] = {
                "fitness": bic
            }
        fitness_values = torch.tensor([self.K[k]["fitness"] for k in self.K] , dtype=torch.float64)
        fitness_values = (fitness_values - fitness_values.mean())/fitness_values.std()
        self.pop_prob = torch.softmax(-fitness_values, dim=0) # low BIC means highest probability
        

    
    def update_obs(self, train_x, train_y):
        self.train_x = train_x.to(self.device)
        self.train_y = train_y.to(self.device)
        obs = list(zip(self.train_x.tolist(), self.train_y.tolist()))
        obs = "\n".join([f"x = {x}, y = {y}" for x, y in obs])
        return obs

    
    def update_system_prompt(self, train_x, train_y):
        obs = self.update_obs(train_x, train_y)
        op = self.operators
        bk = self.base_kernels
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(observations = obs, base_kernels = bk, operators = op)
    

    def parse_kernel(self, expression, d):
        # set of base kernels
        base_kernels = {
            'SE': RBFKernel(ard_num_dims=d),
            'PER': PeriodicKernel(ard_num_dims=d),
            'LIN': LinearKernel(ard_num_dims=d),
            'RQ': RQKernel(ard_num_dims=d),
            'M1': MaternKernel(nu=0.5, ard_num_dims=d),
            'M3': MaternKernel(nu=1.5, ard_num_dims=d),
            'M5': MaternKernel(nu=2.5, ard_num_dims=d)
        }
        def apply_operation(left, op, right):
            if op == '+':
                return left + right
            elif op == '*':
                return left * right

        def parse_subexpression(subexpr):
            base_kernel = re.findall(r'[\w]+', subexpr)
            operators = re.findall(r'[\+\*]', subexpr)

            result = base_kernels[base_kernel[0]] # base kernel
            # apply the operators to the base kernels
            for i, op in enumerate(operators):
                result = apply_operation(result, op, base_kernels[base_kernel[i + 1]])
            return ScaleKernel(result)

        pattern = r'\(([^()]+)\)'
        cache = {} # cache the parsed subexpressions
        while '(' in expression:
            for subexpr in re.findall(pattern, expression):
                if subexpr not in cache:
                    sub_kernel = parse_subexpression(subexpr)
                    cache[subexpr] = sub_kernel
                    base_kernels[f'SubKernel{len(base_kernels)}'] = sub_kernel
                expression = expression.replace(f'({subexpr})', f'SubKernel{len(base_kernels) - 1}', 1)
        return parse_subexpression(expression)

    @staticmethod
    def parse_response(response):
        kernel_start = response.find("Kernel: ") + len("Kernel: ")
        kernel_end = response.find("\n", kernel_start)
        kernel = response[kernel_start:kernel_end]

        analysis_start = response.find("Analysis: ") + len("Analysis: ")
        analysis = response[analysis_start:]
        return kernel, analysis
    
    
    def gp_surrogate(self, train_x, train_y, kernel, device = "cpu"):
        max_iters = 500
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        d = train_x.shape[-1]
        covar_module = self.parse_kernel(kernel, d)
        model = SingleTaskGP(
            train_X= train_x,
            train_Y= train_y.unsqueeze(-1),
            covar_module= covar_module,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=d)
        )
        likelihood = model.likelihood
        likelihood.noise = 1e-4
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # fit the GP model
        fit_gpytorch_mll(mll)

        # calculate BIC
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            output = model(train_x)
            log_likelihood = mll(output, train_y).item()

        num_params = sum(param.numel() for param in model.parameters())
        num_data = train_x.size(0)
        bic = -2 * log_likelihood + num_params * math.log(num_data)

        return model, likelihood, bic



    def sample_parents(self):
        population = list(self.K.keys())
        k1, k2 = np.random.choice(population, size = 2, p= self.pop_prob, replace= False)
        return k1, k2
    
    def get_k_star(self):
        return max(self.K, key = lambda x: self.K[x]["fitness"])

    def cross_over(self):
        for i in range(self.n_c):
            k1, k2 = self.sample_parents()
            try:
                response = self.llm.generate(
                    CROSSOVER_PROMPT_TEMPLATE.format_map(
                        {
                            "parent_kernel1": k1,
                            "parent_kernel2": k2,
                            "fitness1": self.K[k1]["fitness"],
                            "fitness2": self.K[k2]["fitness"],
                            "operators": self.operators
                        }
                    ),
                    self.system_prompt
                )
                k_c, anal = self.parse_response(response)
                print(anal)
                print(k_c)
            except:
                k_c = f"{k1}{np.random.choice(self.operators)}{k2}"
            try:
                model, likelihood, bic = self.gp_surrogate(self.train_x, self.train_y, kernel = k_c, 
                                                           device = self.device)
                self.K[k_c] = {
                    "fitness": bic,
                }
            except:
                continue


    def mutate(self):
        if np.random.rand() < self.p_m:
            k_fittest = self.get_k_star()
            try:
                response = self.llm.generate(MUTATION_PROMPT_TEMPLATE.format_map(
                    {
                        "kernel": k_fittest,
                        "fitness" : self.K[k_fittest]["fitness"],
                        "base_kernels": self.base_kernels
                    }
                ), 
                self.system_prompt
                )
                k_m, _ = self.parse_response(response)
                model, likelihood, bic = self.gp_surrogate(self.train_x, self.train_y, kernel = k_m, 
                                                           device = self.device)
                self.K[k_m] = {
                    "fitness": bic
                }
            except:
                pass


    def keep_top_k(self):
        self.K = dict(sorted(self.K.items(), key = lambda x : x[1]["fitness"]))
        self.K = dict(list(self.K.items())[:self.n_p])
        # Note that unlike the initialisation step, the BIC scores are not normalised after adding new kernels through GA
        self.pop_prob = torch.softmax(torch.tensor([self.K[k]["fitness"] for k in self.K], dtype= torch.float64), dim = 0)

    # def next_point():
    #     pass

    def run(self, train_x, train_y):
        # inside a loop from 1 to T:
        self.update_system_prompt(train_x, train_y)
        self.evaluate_fitness()
        self.cross_over()
        print("Hi 3")
        self.mutate()
        print("Hi 4")
        self.keep_top_k()
        k_star = self.get_k_star()
        print(k_star)
        return k_star
        # train_x, train_y = self.next_point()
        

        

