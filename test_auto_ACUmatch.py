import os
import openai
from datasets import load_dataset
import json
from tabulate import tabulate
import numpy as np
import tqdm
from secret_keys import OPENAI_API_KEY

cnndm_test = load_dataset("Salesforce/rose", "cnndm_test")["data"]

# get openai api key from system
openai.api_key = OPENAI_API_KEY
# %%
def is_ACU_contained_in_summary(summary: str, ACU: str, model: str = 'gpt-4') -> bool:
	message_content = f"""Please determine if the key fact is contained in the article. For the key fact to be considered to be contained, all details of the key fact must be present in the article. For example, the key fact of "A hates B" is **not contained** in the article "A hangout with B and they left unhappily." The data is shown here:\n{{"article": "{summary}", "key fact": "{ACU}"}}\nNow give your answer, only answer Yes or No."""
	completion = openai.ChatCompletion.create(
		model=model,
		messages=[
			{"role": "system",
			 "content": "You are a helpful assistant that diligently check all the details and make factual and objective judgements for the task."},
			{"role": "user", "content": message_content}
		],
		temperature=0,
		max_tokens=10,
	)
	result = completion.choices[0].message.content.lower()
	assert 'yes' in result or 'no' in result
	return 'yes' in result


example_summary_data = \
	{
		"source": "Club Tijuana star Juan Arango conjured memories Luis Suarez in his team's 4-3 defeat by Monterrey in the Mexican league - but it was not through prodigious scoring. The Venezuelan icon Arango sank his teeth into the shoulder of Jesus Zavela as his temper flared in the defeat. He was not booked by the referee but could face a heavy retrospective ban. Juan Arango (left) bites the shoulder of opponent Jesus Zavela in a moment of madness . Zavala holds his shoulder after being bitten by Arango, in the game Zavala's side won 4-3 in Mexico . Zavala shows the referee the mark on his shoulder after being bittern by Arango . Arango (right) earlier scored a magnificent free kick to bring his Club Tijuana team level against Monterrey . Arango had earlier curled in a magnificent free kick for his team to bring them level after falling 2-0 down early on in the encounter. But the 34-year-old overshadowed his goal with the bite as television cameras picked up the moment of madness. Arango spent 10 years playing in Europe, spending five seasons each at Real Mallorca in Spain and Borussia Monchengladbach in Germany. He has made 121 appearances for Venezuela.",
		"reference": "Juan Arango escaped punishment from the referee for biting Jesus Zavela .\nHe could face a retrospective punishment for the incident .\nArango had earlier scored a free kick in his team's 4-3 defeat .",
		"reference_acus": [
			"Juan Arango escaped punishment.",
			"Juan Arango bit Jesus Zavela.",
			"Juan Arango escaped punishment from the referee",
			"Arango could face a retrospective punishment.",
			"Arango could face a retrospective punishment for the incident",
			"Arango had eariler scored a free kick.",
			"Arango scored for his team.",
			"Arango had a defeat.",
			"Arango's team suffered a defeat",
			"Arango's team suffered a 4-3 defeat"
		],
		"count_id": 0,
		"example_id": "000571afe702684d90c1d222ce70b1e1375c1016",
		"annotations": {
			"bart": {
				"acu_labels": [
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1
				],
				"acu": 1.0,
				"normalized_acu": 0.5503382086753845
			},
			"gold": {
				"acu_labels": [
					0,
					1,
					0,
					1,
					1,
					0,
					0,
					0,
					0,
					0
				],
				"acu": 0.30000001192092896,
				"normalized_acu": 0.2684518098831177
			},
			"pegasus": {
				"acu_labels": [
					0,
					0,
					0,
					1,
					0,
					1,
					1,
					0,
					0,
					0
				],
				"acu": 0.30000001192092896,
				"normalized_acu": 0.30000001192092896
			},
			"brio": {
				"acu_labels": [
					0,
					1,
					0,
					1,
					1,
					1,
					1,
					1,
					1,
					1
				],
				"acu": 0.800000011920929,
				"normalized_acu": 0.6230406165122986
			},
			"gsum": {
				"acu_labels": [
					0,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1
				],
				"acu": 0.8999999761581421,
				"normalized_acu": 0.5933165550231934
			},
			"simcls": {
				"acu_labels": [
					0,
					1,
					0,
					1,
					1,
					1,
					1,
					0,
					0,
					0
				],
				"acu": 0.5,
				"normalized_acu": 0.42324087023735046
			},
			"cliff": {
				"acu_labels": [
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1
				],
				"acu": 1.0,
				"normalized_acu": 0.7788007855415344
			},
			"ctrlsum": {
				"acu_labels": [
					0,
					1,
					1,
					1,
					1,
					1,
					0,
					0,
					1,
					1
				],
				"acu": 0.699999988079071,
				"normalized_acu": 0.5015718936920166
			},
			"frost": {
				"acu_labels": [
					0,
					1,
					0,
					0,
					0,
					1,
					1,
					0,
					0,
					0
				],
				"acu": 0.30000001192092896,
				"normalized_acu": 0.30000001192092896
			},
			"glob": {
				"acu_labels": [
					0,
					1,
					1,
					1,
					1,
					0,
					0,
					0,
					0,
					0
				],
				"acu": 0.4000000059604645,
				"normalized_acu": 0.3680177628993988
			},
			"matchsum": {
				"acu_labels": [
					0,
					1,
					0,
					0,
					0,
					0,
					0,
					1,
					1,
					1
				],
				"acu": 0.4000000059604645,
				"normalized_acu": 0.33859267830848694
			},
			"brio-ext": {
				"acu_labels": [
					0,
					1,
					0,
					1,
					1,
					1,
					1,
					0,
					0,
					0
				],
				"acu": 0.5,
				"normalized_acu": 0.39484646916389465
			}
		},
		"system_outputs": {
			"bart": "Juan Arango bites Jesus Zavela in a moment of madness in Club Tijuana's 4-3 defeat by Monterrey in the Mexican league. The Venezuelan icon sank his teeth into the shoulder of the opponent as his temper flared in the defeat. He was not booked by the referee but could face a heavy retrospective ban. Arango had earlier curled in a magnificent free kick for his team to bring them level after falling 2-0 down.",
			"gold": "Juan Arango bit Jesus Zavala's shoulder in Club Tijuana's 4-3 defeat by Monterrey. The Venezuelan icon sank his teeth into the shoulder of Jesus Zavela. Arango was not booked by the referee but could face a heavy retrospective ban.",
			"pegasus": "Club Tijuana lost 4-3 to Monterrey in the Mexican league. Juan Arango was not booked but could face a heavy retrospective ban. A Arango free kick had brought his team level at 2-2.",
			"brio": "Juan Arango bites the shoulder of opponent Jesus Zavela in the Mexican league. The Club Tijuana star earlier scored a magnificent free kick to bring his team level against Monterrey\u00a0. The Venezuelan icon could face a retrospective ban for the bite. Arango's side lost the game 4-3 in Mexico.",
			"gsum": "Juan Arango bites Jesus Zavela in Club Tijuana's 4-3 defeat by Monterrey in the Mexican league. The Venezuelan icon sank his teeth into the shoulder of Jesus Zavala in a moment of madness. He was not booked by the referee but could face a heavy retrospective ban. The 34-year-old had earlier scored a magnificent free kick to bring his team level.",
			"simcls": "Club Tijuana lost 4-3 to Monterrey in the Mexican league on Saturday. Juan Arango bit Jesus Zavela in a moment of madness. The Venezuelan icon could face a heavy retrospective ban. Arango had earlier scored a magnificent free kick to bring his team level.",
			"cliff": "Juan Arango bites Jesus Zavela in Tijuana's 4-3 defeat by Monterrey. The Venezuelan icon sank his teeth into the shoulder of the opponent. He was not booked by the referee but could face a retrospective ban. Arango had earlier scored a magnificent free kick to bring his team level\u00a0.",
			"ctrlsum": "Juan Arango bit the shoulder of opponent Jesus Zavela in Club Tijuana's 4-3 defeat by Monterrey in the Mexican league. He was not booked by the referee but could face a heavy retrospective ban. Arango had earlier scored a magnificent free kick to bring his team level. CLICK HERE for all the latest Mexico news .",
			"frost": "Juan Arango sank his teeth into the shoulder of Jesus Zavela . Club Tijuana lost 4-3 to Monterrey in the Mexican league . Arango earlier scored a magnificent free kick to bring his team level .",
			"glob": "Juan Arango bit Jesus Zavela in Club Tijuana's 4-3 defeat by Monterrey. The Venezuelan icon sank his teeth into the shoulder of Jesus Zavela. He was not booked by the referee but could face a heavy retrospective ban.",
			"matchsum": "Club Tijuana star Juan Arango conjured memories Luis Suarez in his team's 4-3 defeat by Monterrey in the Mexican league - but it was not through prodigious scoring. Juan Arango (left) bites the shoulder of opponent Jesus Zavela in a moment of madness.",
			"brio-ext": "He was not booked by the referee but could face a heavy retrospective ban. Juan Arango (left) bites the shoulder of opponent Jesus Zavela in a moment of madness. Arango (right) earlier scored a magnificent free kick to bring his Club Tijuana team level against Monterrey."
		}
	}
# %%

datapoint = cnndm_test[0]
for model_name in datapoint['system_outputs'].keys():
	# model_name = 'frost'
	# for each summary, evaluate if LLM can correctly identify if ACU is contained inside the summary
	summary_text = datapoint['system_outputs'][model_name]
	ACU_texts = datapoint['reference_acus']

	# compare LLM results with human annotations, calculate precision and recall
	human_results = datapoint['annotations'][model_name]['acu_labels']  # 1 if contained, 0 otherwise
	llm_results = [0 for _ in range(10)]
	for i in range(10):
		ACU_text = ACU_texts[i]
		llm_result = is_ACU_contained_in_summary(summary_text,
												 ACU_text, model='gpt-4')  # True if LLM think ACU is contained in summary, False otherwise
		llm_results[i] = int(llm_result)

		if llm_result == 1 and human_results[i] == 0:
			print('summary: ', summary_text)
			print('ACI', ACU_text)
			print()


	def calculate_precision_and_recall(ground_truth, hypothesis):
		if len(ground_truth) != len(hypothesis):
			raise ValueError("Length of ground_truth and hypothesis must be the same.")

		true_positive = 0
		false_positive = 0
		false_negative = 0

		for gt, pred in zip(ground_truth, hypothesis):
			if gt == 1 and pred == 1:
				true_positive += 1
			if gt == 0 and pred == 1:
				false_positive += 1
			if gt == 1 and pred == 0:
				false_negative += 1

		if true_positive + false_positive == 0:
			precision = 1  # Precision is undefined, can also set to 0 or handle separately
		else:
			precision = true_positive / (true_positive + false_positive)

		if true_positive + false_negative == 0:
			recall = 1  # Recall is undefined, can also set to 0 or handle separately
		else:
			recall = true_positive / (true_positive + false_negative)

		return precision, recall


	print('model_name:', model_name, calculate_precision_and_recall(human_results, llm_results))


# %%
print()
# gpt3.5 turbo result 1
# model_name: bart (1.0, 1.0)
# model_name: gold (0.42857142857142855, 1.0)
# model_name: pegasus (0.375, 1.0)
# model_name: brio (1.0, 0.875)
# model_name: gsum (0.8888888888888888, 0.8888888888888888)
# model_name: simcls (0.7142857142857143, 1.0)
# model_name: cliff (1.0, 0.9)
# model_name: ctrlsum (0.7777777777777778, 1.0)
# model_name: frost (0.42857142857142855, 1.0)
# model_name: glob (0.5714285714285714, 1.0)
# model_name: matchsum (0.5714285714285714, 1.0)
# model_name: brio-ext (0.7142857142857143, 1.0)