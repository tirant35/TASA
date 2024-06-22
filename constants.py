PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

TEST_DIC = {
    "medqa":    {
        "instruction": "The following are multiple choice questions (with answers) about medicine.\n\nQuestion: A 33-year-old female presents to her primary care physician complaining of heat intolerance and difficulty sleeping over a one month period. She also reports that she has lost 10 pounds despite no changes in her diet or exercise pattern. More recently, she has developed occasional unprovoked chest pain and palpitations. Physical examination reveals a nontender, mildly enlarged thyroid gland. Her patellar reflexes are 3+ bilaterally. Her temperature is 99\u00b0F (37.2\u00b0C), blood pressure is 135/85 mmHg, pulse is 105/min, and respirations are 18/min. Laboratory analysis is notable for decreased TSH. Which of the following pathophysiologic mechanisms contributed to the cardiovascular symptoms seen in this patient?\nA. Increased numbers of a1-adrenergic receptors\nB. Decreased numbers of a1-adrenergic receptors\nC. Decreased numbers of a2-adrenergic receptors\nD. Decreased sensitivity of \u00df2-adrenergic receptors\nE. Increased sensitivity of \u00df1-adrenergic receptors\nanswer:",
        "input": "",
        "output": "E",
        "source": "step1"
    },
    "legal":    {
        "input": "Clause: It is important that you protect and maintain the security of your Account and that you immediately notify us of any unauthorized use of your Account.\nQuestion: is my information secure",
        "output": "Relevant",
        "instruction": "Classify if the clause is relevant to answering the question by responding with Relevant if the clause provides useful information for answering the question, or Irrelevant if there is no direct connection between the clause and the question."
    },
    "imdb":        {
        "input": "It seems like more consideration has gone into the IMDb reviews of this film than went into the source.<br /><br />Here's a review without pretensions:<br /><br />Just when you think nothing is going to happen, it doesn't.<br /><br />Dress it up any way you like, this is a dull film, full of unengaging characters doing very little of interest.<br /><br />One to put on if you want to convince an impressionable emo chick that you're like, so deep, man.<br /><br />Not something to watch for your own pleasure though.<br /><br />Unless.<br /><br />You're.<br /><br />Pretentious.",
        "output": "neg",
        "instruction": "A mark is positive when the review expresses favorable or approving sentiment.A mark is negative when the review conveys an unfavorable or disapproving opinion.A mark is unsupervised if the review lacks a clear opinion or if the opinion is ambiguous.Label the type of mark for the following review."
    }
}

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"