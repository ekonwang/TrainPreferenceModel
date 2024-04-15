import os, sys
import time
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig

MSG = "The primary goal of effective pest control for Kennett Square is to offer you with optimum comfort and safety and security for yourself and also your entire family. With complete pest control Kennett Square services, our team believe in safe, non-chemical, safe treatments, so back them up with ensured satisfaction.\nThe most effective means to see to it that your home is without insects and also rodents is by calling our company specializing in this area. It is best to find out as much info as feasible concerning every one prior to you hire them.\nCompanies offering pest control services in Kennett Square usually have a group of experienced professionals that focus on the different sorts of pests that live and reproduce in your home. They are trained on what to do, exactly how to do it, when, as well as where to do it. Some of the most typical insects treated are termites, bed bugs, roaches, and rats.\nEliminating termites and various other pests are very vital for your family members’ health, comfort, and also possessions.\nWe Treat The Following Pest Problems in Kennett Square:\nWhy Choose Our Local Kennett Square Pest control specialists\n- Full range of pest control and termite treatment services.\n- Flexible scheduling, as well as the same and next-day pest control service.\n- Using the latest in pest control and inspection technology for business owners.\n- Highly trained and experienced representatives.\n- Providing 24/7 support as well as emergency services.\n- Green, environmentally-friendly treatments that are child and pet-friendly.\nBed Bug Exterminator Near Me\nWhen a bedbug infestation is identified, you require to get in touch with a certified bedbug exterminator for complete comfort. Knowing that their group of experts is working with getting rid of any type of prospective infestations. Do It Yourself services to pest management are usually ineffective at removing bed bugs due to these blood-sucking insects’ resistance and strength against traditional pesticide treatments.\nThis is why pest management experts in Kennett Square are always available to help exterminate this persistent pest and keep your home or home pest-free. One of the typical ways to control bugs is to use pesticides as well as lures.\nA bedbug exterminator can make use of numerous techniques to get rid of bugs that do not react to traditional pesticides and lures. One of the most typical pest management method is to get rid of your pest issue and all the infestations. This is why it is necessary to get in touch with a pest management professional quickly.\nSpeak With a Plumbing Expert Today!\nPest Control Services Near Me\nDo you have roaches or mosquitoes? Do you need a pest-free home? You require to recognize the nature of pests too. Yet if you have a pest infestation in your house or workplace, you need to think about employing a Kennett Square professional in this area.\nOur Kennett Square pest control experts offer several services for both residential and also commercial structures, consists of:\n- Emergency Pest Control\n- Cockroach Exterminator\n- Ant Control\n- Rodent control\n- Wasp and Bee Removal\n- Termite Control & Treatment\n- Bed Bug Treatment\n- Residential and Commercial Exterminators\n- Spider Control\n- Mosquito control\n- Bat Extermination\n- Tick and Flea Control\n- And Much More!\nEmergency Pest Control in Kennett Square, PA\nFor those who have had experience in pest control and also pest inspection, the idea of mosting likely to an emergency pest control or a Kennett Square pest inspection company can be rather upsetting. If you are uncertain where to start your search, the internet can be a wonderful location to begin, as well as there are many sources available to help you find the best service for you.\nWhen selecting an emergency pest control company in Kennett Square, PA, remember that you will be charged based upon how severe the problem is. For instance, if a small insect infestation has created damage to your home as well as you want to be eliminate them immediately, you will possibly pay a fee that is a lot more than simply a little of cash.\nKennett Square Residential Pest Control\nIf you are a local of an apartment building or a townhome area, you need to be aware that there is a severe possibility for a homeowner to find throughout termites, roaches, ants, wood-destroying insects, bed bugs, and also many other sorts of rodents if they fail to provide proper residential pest control Kennett Square, Pennsylvania. Yes, residential professional pest control services will consist of examining your house for possible termites, removing active pests, and also stopping future infestations from taking place.\nIn many cases, the residential exterminator that you employ will certainly also make use of details sorts of chemicals to get rid of undesirable pests and also rodents. There is nothing even worse than not having a complete termite control and rodent control strategy in place for your apartment building or townhome neighborhood.\nResidential pest control can be done by a professional Kennett Square exterminator that specializes in pest control, or you can call your local exterminator and request for their suggestions. There is no reason that you can not obtain pest control today to help you with your pest issue.\nSpeak With a Plumbing Expert Today!\nCommercial Pest Control Pennsylvania for Businesses\nKennett Square Commercial pest management is essential for businesses that offer services to the public. They require to secure their food production, inventory, and also various other supplies. They must additionally be aware of any kind of problems related to pests that may be influencing their food.\nA company owner should hire our Kennett Square company to come bent on inspect and also control pests for them. A pest inspection report from a professional company in Kennett Square can help proprietors as well as supervisors understand what problems may be present as well as address those problems. Businesses should watch out for obtaining pest inspections done by themselves. It is important to make sure the inspection is done by a professional.\nOur team of qualified experts will certainly collaborate with you to develop an effective management plan for your business as well as your procedures and also pest management services.\nComplete Pest Management in Kennett Square, PA\nPest management is something that numerous homeowners think about, yet few in fact do. Pest control has its advantages, yet it likewise has its costs. It can be a very pricey task to have actually a professional come out as well as spray for months. It can additionally cost a fair bit of cash to hire somebody to eliminate the pests once they are already in your home. This is why many people pick to benefit from eco-friendly pest control in Kennett Square.\nThey will also give you ideas on utilizing all-natural products to get rid of pests as well as exactly how to stop pest infestations in your Kennett Square, PA home or business. From occurring again in the future.\nThere are several eco-friendly pest management companies that you can make use of, so make certain that you make the appropriate choice– a company with a positive credibility for helping people get rid of pests safely as well as efficiently."
PREF_HF = "/fs-computility/llm/shared/wangyikun/ckpts/internlm2-preference-V1_0-1_8b"
model = AutoModel.from_pretrained(PREF_HF, trust_remote_code=True)
device = 'cuda'
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(PREF_HF, trust_remote_code=True)
token_ids = tokenizer(MSG, return_tensors='pt')
# expand along the first axis
for k in token_ids.keys():
    token_ids[k] = token_ids[k][:, :512].expand(64, -1).to(device)

start_time = time.time()
with torch.no_grad():
    outputs = model(**token_ids)
eclipsed = time.time() - start_time
print(f"Time taken: {eclipsed:.3f} s, throughout: {64 / eclipsed:.3f} sents/s")
import pdb; pdb.set_trace()
