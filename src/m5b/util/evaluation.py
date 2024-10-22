import re
from typing import Dict, List, Literal, Set

import evaluate
import numpy as np
import pandas as pd


def generated_caption_evaluation(
    caption_preds: List[str],
    caption_golds: List[str] | List[List[str]],
    lang_id: str,
    metrics: Set[str] | None = None,
    use_gpu: bool = True,
) -> Dict[str, float | str]:
    if metrics is None:
        metrics = {"bleu", "meteor", "rouge", "bertscore", "chrf"}
        # "CIDEr", "SPICE",

    if isinstance(caption_golds[0], list):
        # When we have multiple gold captions, ChrF, as implemented by sacrebleu,
        # requires the same number of references for each prediction
        max_len = max(len(x) for x in caption_golds)
        for i in range(len(caption_golds)):
            caption_golds[i] = caption_golds[i] + [""] * (
                max_len - len(caption_golds[i])
            )

    results = dict()
    if "bleu" in metrics:
        bleu = evaluate.load("bleu")
        for bleu_n in [1, 2, 3, 4]:
            res = bleu.compute(
                predictions=caption_preds,
                references=caption_golds,
                max_order=bleu_n,
            )
            results[f"bleu_{bleu_n}"] = res["bleu"]
    if "rouge" in metrics:
        rouge = evaluate.load("rouge")
        res = rouge.compute(
            predictions=caption_preds,
            references=caption_golds,
        )
        for rouge_method, rouge_res in res.items():
            results[rouge_method] = rouge_res
    if "meteor" in metrics:
        meteor = evaluate.load("meteor")
        res = meteor.compute(
            predictions=caption_preds,
            references=caption_golds,
        )
        results["meteor"] = res["meteor"]
    if "chrf" in metrics:
        chrf = evaluate.load("chrf")

        for m, wo in {"": 0, "+": 1, "++": 2}.items():
            res = chrf.compute(
                predictions=caption_preds,
                references=caption_golds,
                word_order=wo,
            )
            results[f"chrF{m}"] = res["score"]
    if "bertscore" in metrics:
        bertscore = evaluate.load("bertscore", device="cuda:0" if use_gpu else "cpu")
        if lang_id == "fil":
            lang_id = "tl"
        elif lang_id == "quz":
            lang_id = "qu"
        res = bertscore.compute(
            predictions=caption_preds,
            references=caption_golds,
            lang=lang_id,
            device="cuda:0" if use_gpu else "cpu",
        )
        for r in ["precision", "recall", "f1"]:
            results[f"avg_bertscore_{r}"] = np.mean(res[r])
            results[f"std_bertscore_{r}"] = np.std(res[r])
    if "cider" in metrics or "spice" in metrics:
        raise NotImplementedError(
            "CIDEr and SPICE are not supported by the current implementation"
        )
    return results


def generated_label_classification_evaluation(
    gold_labels: List[str] | List[List[str]],
    pred_labels: List[str],
    vqa_post_process: bool = True,
    bool_to_yes_no: bool = True,
    entailment_to_yes_no_maybe: bool = False,
    remove_trailing_period: bool = True,
) -> Dict[str, float]:
    single_answers = isinstance(gold_labels[0], str)
    print(f"Evaluation with {single_answers=}")

    df = pd.DataFrame(
        {
            "gold": gold_labels,
            "pred": pred_labels,
        }
    )
    if single_answers:
        df["gold"] = df["gold"].apply(lambda x: str(x).strip())
        if remove_trailing_period:
            df["pred"] = df["pred"].apply(lambda x: str(x).rstrip("."))
    else:
        df["gold"] = df["gold"].apply(lambda x: [str(ans).strip() for ans in x])
        if remove_trailing_period:
            df["pred"] = df["pred"].apply(lambda x: [str(ans).rstrip(".") for ans in x])

    if entailment_to_yes_no_maybe:

        def _entailment_to_yes_no_maybe(x) -> Literal["yes", "no", "maybe"]:
            if x.lower() in ["yes", "no", "maybe"]:
                return x.lower()
            elif x.lower() in ["true", "false"]:
                return "yes" if x.lower() == "true" else "no"
            elif x.lower() in ["1", "0"]:
                return "yes" if x.lower() == "1" else "no"
            elif x.lower() in ["entailment", "contradiction", "neutral"]:
                if x.lower() == "entailment":
                    return "yes"
                elif x.lower() == "contradiction":
                    return "no"
                else:
                    return "maybe"
            else:
                return x

        df["pred"] = df["pred"].apply(_entailment_to_yes_no_maybe)
        if single_answers:
            df["gold"] = df["gold"].apply(_entailment_to_yes_no_maybe)
        else:
            df["gold"] = df["gold"].apply(
                lambda x: [_entailment_to_yes_no_maybe(ans) for ans in x]
            )

    if bool_to_yes_no:

        def map_bool_to_yes_no(x):
            if isinstance(x, bool):
                return "yes" if x else "no"
            elif isinstance(x, str):
                if x.lower() == "true":
                    return "yes"
                elif x.lower() == "false":
                    return "no"
            return x

        df.pred = df.pred.apply(lambda x: str(map_bool_to_yes_no(x)).strip())
        if single_answers:
            df.gold = df.gold.apply(lambda ans: str(map_bool_to_yes_no(ans)).strip())
        else:
            df.gold = df.gold.apply(
                lambda answers: [map_bool_to_yes_no(ans) for ans in answers]
            )

    df["pred_post_processed"] = vqa_clean(df["pred"].tolist())

    scores: Dict[str, float] = {}
    if single_answers:
        scores["acc"] = (df["gold"] == df["pred"]).mean()
        scores["relaxed_acc"] = df.apply(
            lambda x: x["pred"].startswith(x["gold"]) or x["pred"].endswith(x["gold"]),
            axis=1,
        ).mean()

        if vqa_post_process:
            post_proc_acc = (df["gold"] == df["pred_post_processed"]).mean()
            post_proc_relaxed_acc = df.apply(
                lambda x: x["pred_post_processed"].startswith(x["gold"])
                or x["pred_post_processed"].endswith(x["gold"]),
                axis=1,
            ).mean()
            scores["acc_post_processed"] = post_proc_acc
            scores["relaxed_acc_post_processed"] = post_proc_relaxed_acc
    else:
        scores["acc"] = df.apply(lambda x: x["pred"] in x["gold"], axis=1).mean()
        scores["relaxed_acc"] = df.apply(
            lambda x: any(
                x["pred"].startswith(ans) or x["pred"].endswith(ans)
                for ans in x["gold"]
            ),
            axis=1,
        ).mean()

        if vqa_post_process:
            scores["acc_post_processed"] = df.apply(
                lambda x: any(x["pred_post_processed"] == ans for ans in x["gold"]),
                axis=1,
            ).mean()
            scores["relaxed_acc_post_processed"] = df.apply(
                lambda x: any(
                    x["pred_post_processed"].startswith(ans)
                    or x["pred_post_processed"].endswith(ans)
                    for ans in x["gold"]
                ),
                axis=1,
            ).mean()

    return scores


# adapted from https://github.com/salesforce/LAVIS/blob/main/lavis/common/vqa_tools/vqa_eval.py
def vqa_clean(labels: List[str]):
    manualMap = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = ["a", "an", "the"]
    periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile(r"(\d)(,)(\d)")
    punct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(commaStrip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    cleaned_labels = []
    for label in labels:
        label = label.replace("\n", "").replace("\t", "").strip()
        label = processPunctuation(label)
        label = processDigitArticle(label)
        cleaned_labels.append(label)
    return cleaned_labels
