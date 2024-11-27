config = {
    'data_dir' : 'data/MATH/',
    'save_dir' : 'saved_models/',
    'load_dir' : 'saved_models/',

    'policy_model_name' : 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ', 

    'stage_one_prompt' : '[INS] You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct." [/INS]',
    'stage_two_prompt' : '[INS] There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct." [/INS]',
    
    'batch_size' : 2,
    'stage_one_epochs' : 100,
    'stage_two_epochs' : 100,
    'alpha': 10,
    'beta_one': 0.01,
    'beta_two': 0.1,
    'lr': 5e-6,

    'gen_kwargs' : {
            'max_new_tokens': 1000,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': True,
            'temperature': 1.0
        }
}