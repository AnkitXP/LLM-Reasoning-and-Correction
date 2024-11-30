config = {
    'model_dir' : 'base_models/',
    'save_dir' : 'saved_models/',
    'load_dir' : 'saved_models/[Add Complete Path]/',
    'log_dir' : 'logs/',

    'policy_model_name' : 'ai-nexuz/llama-3.2-1b-instruct-fine-tuned',

    'seed' : 44,
    'batch_size' : 2,
    'local_rollout_forward_batch_size': 1,
    'gradient_accumulation_steps': 1,
    'total_episodes' : 100,
    'save_interval' : 20,
    'alpha': 10,
    'beta_one': 0.01,
    'beta_two': 0.1,
    'lr': 5e-6,

    'gen_kwargs' : {
            'max_new_tokens': 500,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': True,
            'temperature': 1.0,
            'return_dict_in_generate' : True,
            'output_scores' : True,
        },

    'first_attempt_prompt' : 
    
"""You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct." 

Here is an example:

Example 1:
Problem: Solve for x: 2x + 3 = 7.
Solution:
Step 1: Subtract 3 from both sides to get 2x = 4.
Step 2: Divide both sides by 2 to get x = 2.
Final Answer: The final answer is \\boxed{2}. I hope it is correct.""",


    'second_attempt_prompt' : """There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct." """,

}