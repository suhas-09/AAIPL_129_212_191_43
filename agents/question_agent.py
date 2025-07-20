#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent

import random
import json

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""
    
    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str)->str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            'EXAMPLE: {}\n'
            '{{\n'
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            '}}'            
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += fmt.format(topic, topic.split('/')[-1], question, *choices, answer, explanation) + "\n\n"
        return sample_str.strip()


    def build_prompt(self, topic: str, wadvsys: bool = True, wicl: bool = True, inc_samples: List[Dict[str, str]]|None = None) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""
        
        if wadvsys:
            # TODO: Manipulate this SYS prompt for better results
            sys_prompt = """
You are an expert generator of complex logic puzzles. Your task is to create a multiple-choice question with a single correct answer based on a given topic. Each puzzle must be logically sound, with a detailed step-by-step explanation demonstrating how the solution is reached. The provided options must be clear and concise.

Follow the exact format of the examples below.

Example 1: Knights and Knaves

Topic: Knights, Knaves, and Spies

Scenario:
You meet three suspects in a vault: Aris, Bree, and Cora. You know one is a Knight (always tells the truth), one is a Knave (always lies), and one is a Spy (can do either). They make the following statements:

Aris says: "Bree is the Knave."

Bree says: "Cora is the Knight."

Question:
If the Spy is the only one who knows the true identity of the Knave, who is the Knave?

A) Aris
B) Bree
C) Cora
D) It cannot be determined from the information.

Correct Answer: B) Bree

Explanation:
The key is the condition: "The Spy is the only one who knows the true identity of the Knave."

Analyze Aris's statement: Aris says, "Bree is the Knave." By making this direct claim about the Knave's identity, Aris is claiming to know who the Knave is.

Apply the key condition: Since only the Spy knows the Knave's identity, Aris, by claiming this knowledge, must be the Spy. The Spy can lie or tell the truth, so we don't yet know if Bree is actually the Knave, but we have successfully identified Aris.

Determine the remaining roles: Since Aris is the Spy, Bree and Cora must be the Knight and the Knave, in some order.

Analyze Bree's statement: Now we know Bree is either the Knight or the Knave. Bree says, "Cora is the Knight."

Case 1 (Assume Bree is the Knight): If Bree is the Knight, her statement must be true. This would make Cora the Knight. This is a contradiction, as we can't have two Knights. Therefore, this case is impossible.

Case 2 (Assume Bree is the Knave): If Bree is the Knave, her statement must be false. Her statement is "Cora is the Knight," so the truth is that Cora is not the Knight. Since Cora is not the Knight and not the Spy (Aris is), she must be the Knave. But wait, this is also a contradiction, we can't have two Knaves (Bree and Cora).

Re-evaluate Aris's statement: Let's re-read the deduction. Aris is the Spy. The Spy can lie or tell the truth.

If Aris (the Spy) is telling the truth, then Bree is the Knave. This makes Cora the Knight. Let's check consistency.

Aris (Spy) says "Bree is the Knave." (True)

Bree (Knave) says "Cora is the Knight." This must be a lie. The opposite is "Cora is not the Knight," which is true in this scenario. This works.

If Aris (the Spy) is lying, then Bree is not the Knave. Since Bree is not the Spy either, she must be the Knight. This would make Cora the Knave. Let's check consistency.

Aris (Spy) says "Bree is the Knave." (Lie)

Bree (Knight) says "Cora is the Knight." This must be true. This is a contradiction, as Cora is the Knave in this scenario.

Conclusion: The only consistent scenario is that Aris is the Spy telling the truth. This makes Bree the Knave and Cora the Knight.

Example 2: Family Tree Relations

Topic: Family Tree Logic

Scenario:
Review the following facts about a family:

Tom and Mary have only one child, a son named Sam.

Sam is married to a woman named Jenny.

Jenny's only sibling is named Alex.

Alex is married to a woman named Priya.

Priya and Alex have one child, named Leo.

Question:
What is the relationship of Leo to Tom?

A) Leo is Tom's grandson.
B) Leo is Tom's nephew.
C) Leo is Tom's grandnephew.
D) Leo has no direct relation to Tom.

Correct Answer: C) Leo is Tom's grandnephew.

Explanation:
The goal is to trace the relationship path from Leo back to Tom.

Identify Leo's parents: Fact 5 states Leo's parents are Alex and Priya.

Identify Alex's relationship to the main family: Fact 3 states Alex is Jenny's only sibling.

Identify Jenny's relationship: Fact 2 states Jenny is married to Sam.

Identify Sam's relationship: Fact 1 states Sam is the son of Tom.

Combine the links:

Leo is the son of Alex.

Alex is the brother of Jenny (Sam's wife). This makes Alex the brother-in-law of Sam.

Sam is the son of Tom.

Determine the final relationship: Since Sam is Tom's son, Sam's brother-in-law (Alex) is Tom's son-in-law's brother (or more simply, just a relative by marriage). However, the child of one's nephew or niece is a grandnephew or grandniece. From Tom's perspective, his son is Sam. Sam's wife is Jenny. Jenny's brother is Alex. Therefore, Alex is Tom's son's brother-in-law. While not a blood nephew, the child of this "nephew-in-law" (Alex) is conventionally called a grandnephew.

Eliminate other options:

A grandson (A) would be the son of Tom's child (Sam). Leo is not Sam's son.

A nephew (B) would be the son of Tom's sibling. We don't know if Tom has siblings, but Leo is not the son of one.

No direct relation (D) is incorrect because a clear, albeit complex, family path exists.

Example 3: Seating Arrangements

Topic: Conditional Seating Logic

Scenario:
Six diplomats—Faisal, Grace, Hao, Ivan, Jiya, and Ken—are seated at a large circular table with six chairs.

Faisal is sitting directly opposite Hao.

Grace is sitting two seats to the left of Faisal.

Jiya is not sitting next to Hao.

Ivan is sitting next to Ken, but only if Ken is sitting next to Hao.

Question:
Who is sitting to the immediate right of Grace?

A) Faisal
B) Jiya
C) Hao
D) Ivan

Correct Answer: B) Jiya

Explanation:
The key to this puzzle is the conditional rule (Rule 4).

Place the initial pair: Start with Rule 1. Place Faisal at the top (seat 1) and Hao at the bottom (seat 4).

Seat 1: Faisal

Seat 4: Hao

Place the next person: Use Rule 2. Grace is two seats to the left of Faisal (seat 1). Counting leftward, seat 6 is one left, so seat 5 is two left.

Seat 5: Grace

Analyze the conditional rule: Rule 4 states "Ivan is next to Ken, only if Ken is next to Hao." Let's test the condition: "Is it possible for Ken to sit next to Hao?" Hao is in seat 4, so the adjacent seats are 3 and 5. Seat 5 is already taken by Grace. Therefore, the only place Ken could sit to be next to Hao is seat 3.

Test the hypothesis: Let's assume Ken is in seat 3.

The condition "Ken is sitting next to Hao" is now met.

Therefore, the main part of the rule must be true: "Ivan is sitting next to Ken." Ken is in seat 3, so Ivan must be in seat 2 or 4. Seat 4 is taken by Hao, so Ivan must go in seat 2.

Check for consistency: The current arrangement is:

Seat 1: Faisal

Seat 2: Ivan

Seat 3: Ken

Seat 4: Hao

Seat 5: Grace

Seat 6: ???
The only person left is Jiya, who must be in seat 6.

Verify all rules: We must check this final arrangement against every rule.

Rule 1: Faisal opposite Hao. (1 vs 4) - ✓ Correct.

Rule 2: Grace two seats left of Faisal. (5 vs 1) - ✓ Correct.

Rule 3: Jiya not next to Hao. Jiya is in 6, Hao is in 4. They are not next to each other. - ✓ Correct.

Rule 4: The condition (Ken next to Hao) is met, and the outcome (Ivan next to Ken) is also met. - ✓ Correct.

Answer the question: The arrangement is unique and correct. The question is: "Who is sitting to the immediate right of Grace?" Grace is in seat 5. The seat to her right is seat 6, which is occupied by Jiya.
            """
        else:
            sys_prompt = """
You are an expert generator of complex logic puzzles. Your task is to create a multiple-choice question with a single correct answer based on a given topic. Each puzzle must be logically sound, with a detailed step-by-step explanation demonstrating how the solution is reached. The provided options must be clear and concise.

Follow the exact format of the examples below.

Example 1: Knights and Knaves

Topic: Knights, Knaves, and Spies

Scenario:
You meet three suspects in a vault: Aris, Bree, and Cora. You know one is a Knight (always tells the truth), one is a Knave (always lies), and one is a Spy (can do either). They make the following statements:

Aris says: "Bree is the Knave."

Bree says: "Cora is the Knight."

Question:
If the Spy is the only one who knows the true identity of the Knave, who is the Knave?

A) Aris
B) Bree
C) Cora
D) It cannot be determined from the information.

Correct Answer: B) Bree

Explanation:
The key is the condition: "The Spy is the only one who knows the true identity of the Knave."

Analyze Aris's statement: Aris says, "Bree is the Knave." By making this direct claim about the Knave's identity, Aris is claiming to know who the Knave is.

Apply the key condition: Since only the Spy knows the Knave's identity, Aris, by claiming this knowledge, must be the Spy. The Spy can lie or tell the truth, so we don't yet know if Bree is actually the Knave, but we have successfully identified Aris.

Determine the remaining roles: Since Aris is the Spy, Bree and Cora must be the Knight and the Knave, in some order.

Analyze Bree's statement: Now we know Bree is either the Knight or the Knave. Bree says, "Cora is the Knight."

Case 1 (Assume Bree is the Knight): If Bree is the Knight, her statement must be true. This would make Cora the Knight. This is a contradiction, as we can't have two Knights. Therefore, this case is impossible.

Case 2 (Assume Bree is the Knave): If Bree is the Knave, her statement must be false. Her statement is "Cora is the Knight," so the truth is that Cora is not the Knight. Since Cora is not the Knight and not the Spy (Aris is), she must be the Knave. But wait, this is also a contradiction, we can't have two Knaves (Bree and Cora).

Re-evaluate Aris's statement: Let's re-read the deduction. Aris is the Spy. The Spy can lie or tell the truth.

If Aris (the Spy) is telling the truth, then Bree is the Knave. This makes Cora the Knight. Let's check consistency.

Aris (Spy) says "Bree is the Knave." (True)

Bree (Knave) says "Cora is the Knight." This must be a lie. The opposite is "Cora is not the Knight," which is true in this scenario. This works.

If Aris (the Spy) is lying, then Bree is not the Knave. Since Bree is not the Spy either, she must be the Knight. This would make Cora the Knave. Let's check consistency.

Aris (Spy) says "Bree is the Knave." (Lie)

Bree (Knight) says "Cora is the Knight." This must be true. This is a contradiction, as Cora is the Knave in this scenario.

Conclusion: The only consistent scenario is that Aris is the Spy telling the truth. This makes Bree the Knave and Cora the Knight.

Example 2: Family Tree Relations

Topic: Family Tree Logic

Scenario:
Review the following facts about a family:

Tom and Mary have only one child, a son named Sam.

Sam is married to a woman named Jenny.

Jenny's only sibling is named Alex.

Alex is married to a woman named Priya.

Priya and Alex have one child, named Leo.

Question:
What is the relationship of Leo to Tom?

A) Leo is Tom's grandson.
B) Leo is Tom's nephew.
C) Leo is Tom's grandnephew.
D) Leo has no direct relation to Tom.

Correct Answer: C) Leo is Tom's grandnephew.

Explanation:
The goal is to trace the relationship path from Leo back to Tom.

Identify Leo's parents: Fact 5 states Leo's parents are Alex and Priya.

Identify Alex's relationship to the main family: Fact 3 states Alex is Jenny's only sibling.

Identify Jenny's relationship: Fact 2 states Jenny is married to Sam.

Identify Sam's relationship: Fact 1 states Sam is the son of Tom.

Combine the links:

Leo is the son of Alex.

Alex is the brother of Jenny (Sam's wife). This makes Alex the brother-in-law of Sam.

Sam is the son of Tom.

Determine the final relationship: Since Sam is Tom's son, Sam's brother-in-law (Alex) is Tom's son-in-law's brother (or more simply, just a relative by marriage). However, the child of one's nephew or niece is a grandnephew or grandniece. From Tom's perspective, his son is Sam. Sam's wife is Jenny. Jenny's brother is Alex. Therefore, Alex is Tom's son's brother-in-law. While not a blood nephew, the child of this "nephew-in-law" (Alex) is conventionally called a grandnephew.

Eliminate other options:

A grandson (A) would be the son of Tom's child (Sam). Leo is not Sam's son.

A nephew (B) would be the son of Tom's sibling. We don't know if Tom has siblings, but Leo is not the son of one.

No direct relation (D) is incorrect because a clear, albeit complex, family path exists.

Example 3: Seating Arrangements

Topic: Conditional Seating Logic

Scenario:
Six diplomats—Faisal, Grace, Hao, Ivan, Jiya, and Ken—are seated at a large circular table with six chairs.

Faisal is sitting directly opposite Hao.

Grace is sitting two seats to the left of Faisal.

Jiya is not sitting next to Hao.

Ivan is sitting next to Ken, but only if Ken is sitting next to Hao.

Question:
Who is sitting to the immediate right of Grace?

A) Faisal
B) Jiya
C) Hao
D) Ivan

Correct Answer: B) Jiya

Explanation:
The key to this puzzle is the conditional rule (Rule 4).

Place the initial pair: Start with Rule 1. Place Faisal at the top (seat 1) and Hao at the bottom (seat 4).

Seat 1: Faisal

Seat 4: Hao

Place the next person: Use Rule 2. Grace is two seats to the left of Faisal (seat 1). Counting leftward, seat 6 is one left, so seat 5 is two left.

Seat 5: Grace

Analyze the conditional rule: Rule 4 states "Ivan is next to Ken, only if Ken is next to Hao." Let's test the condition: "Is it possible for Ken to sit next to Hao?" Hao is in seat 4, so the adjacent seats are 3 and 5. Seat 5 is already taken by Grace. Therefore, the only place Ken could sit to be next to Hao is seat 3.

Test the hypothesis: Let's assume Ken is in seat 3.

The condition "Ken is sitting next to Hao" is now met.

Therefore, the main part of the rule must be true: "Ivan is sitting next to Ken." Ken is in seat 3, so Ivan must be in seat 2 or 4. Seat 4 is taken by Hao, so Ivan must go in seat 2.

Check for consistency: The current arrangement is:

Seat 1: Faisal

Seat 2: Ivan

Seat 3: Ken

Seat 4: Hao

Seat 5: Grace

Seat 6: ???
The only person left is Jiya, who must be in seat 6.

Verify all rules: We must check this final arrangement against every rule.

Rule 1: Faisal opposite Hao. (1 vs 4) - ✓ Correct.

Rule 2: Grace two seats left of Faisal. (5 vs 1) - ✓ Correct.

Rule 3: Jiya not next to Hao. Jiya is in 6, Hao is in 4. They are not next to each other. - ✓ Correct.

Rule 4: The condition (Ken next to Hao) is met, and the outcome (Ivan next to Ken) is also met. - ✓ Correct.

Answer the question: The arrangement is unique and correct. The question is: "Who is sitting to the immediate right of Grace?" Grace is in seat 5. The seat to her right is seat 6, which is occupied by Jiya.
            """
        tmpl = (
            'Generate an EXTREMELY DIFFICULT MCQ on topic: {0}.\n\n'

            '**CRITICAL REQUIREMENTS:**\n'
            '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
            '2.  **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.\n'
            '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
            '4.  **Single Correct Answer**: Ensure that option {2} is only factually correct.\n'
            '5.  **Plausible Distractors**: While option {3} are three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic, designed to mislead someone without expert knowledge.\n'
            '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
            '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.\n\n'

            '{5}'
            
            'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
            
            'EXAMPLE: {6}\n'
            '{{\n'
            '  "topic": "{7}",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "Provide a brief explanation why {9} is correct within 100 words."\n'
            '}}'
        )
        # Remove model's preferential bias for options
        correct_option = random.choice(['A', 'B', 'C', 'D'])
        distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(topic, topic, correct_option, distractors, correct_option, inc_samples_ex, topic, topic.split('/')[-1], correct_option, correct_option)

        return prompt, sys_prompt


    def generate_question(self, topic: Tuple[str, str]|List[Tuple[str, str]], wadvsys: bool, wicl: bool, inc_samples: Dict[str, List[Dict[str, str]]]|None, **gen_kwargs) -> Tuple[List[str], int|None, float|None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]])
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]])
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt


    def generate_batches(self, num_questions: int, topics: Dict[str, List[str]], batch_size: int = 5, wadvsys: bool=True, wicl: bool = True, inc_samples: Dict[str, List[Dict[str, str]]]|None = None, **kwargs) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")
        
        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i:i + batch_size]
            batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)
        # for last batch with less than batch_size
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size):]
            batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(self, questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
        def basic_checks(q2: Dict[str, str])->bool:
            # check required keys
            required_keys = ['topic', 'question', 'choices', 'answer']
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
                if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                    # check answer format
                    # Check token length
                    check_len = sum(self.count_tokens_q(q2[k]) for k in ['question', 'answer'])
                    check_len += sum(self.count_tokens_q(choice) for choice in q2['choices']) - 15
                    if check_len < 130:
                        if check_len + self.count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2['answer'], str):
                                return True
            return False
        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()
    
    def save_questions(self, questions: Any, file_path: str|Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(questions, f, indent=4)
    
    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        
        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str|Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples

# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    argparser.add_argument("--num_questions", type=int, default=200, help="Total number of questions to generate.")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file name to save the generated questions.")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for generating questions.")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f: topics = json.load(f)
    
    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics, 
        batch_size=args.batch_size,
        wadvsys=False,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "="*50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n")
        print("\n" + "+"*50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            prompt = (
                'Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n'
                'Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n'
                
                'String:\n'
                '{}\n\n'

                'Given Format:\n'
                '{{\n'
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                '}}'
            )
            q = agent.agent.generate_response(prompt.format(q), "You are an expert JSON extractor.", max_new_tokens=1024, temperature=0.0, do_sample=False)
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
