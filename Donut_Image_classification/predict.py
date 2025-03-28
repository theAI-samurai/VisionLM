import os
import re
from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
import torch
import pandas as pd
from PIL import Image
from typing import Any, List, Tuple


class DonutClassification:

    def __init__(self, model_path, task_prompt):
        self.max_length = 8
        self.image_size = [800, 620]
        self.model_path = model_path
        self.task_prompt = task_prompt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.additional_tokens = [
            '<birth_certificate/>', '<curriculum_Vitae/>', '<decree_on_implementation_order/>',
            '<education_certificate/>', '<family_card/>','<honor_award_decree_and_certificate/>', '<identity_card/>',
            '<indonesian_armed_forces_card/>', '<marriage_certificate/>','<military_education_certificate/>',
            '<passport_size_photo/>','<rank_promotion_decree/>', '<soldier_enrollment_documents/>',
            '<spouse_appointment_letter/>', '<taxpayer_identification_number/>',"<s_class>", "</s_class>", "<s_rvlcdip>"
        ]
        self.model = None
        self.config = None
        self.processor = None
        self.preprocess()
        self.load_model()

    def preprocess(self):
        self.config = VisionEncoderDecoderConfig.from_pretrained("nielsr/donut-base")
        self.config.encoder.image_size = self.image_size  # (height, width)
        self.config.decoder.max_length = self.max_length

        self.processor = DonutProcessor.from_pretrained("nielsr/donut-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("nielsr/donut-base",
                                                               config=self.config,
                                                               ignore_mismatched_sizes=True)
        self.processor.feature_extractor.size = self.image_size[::-1]  # should be (width, height)
        self.processor.feature_extractor.do_align_long_axis = False
        self.add_tokens(self.additional_tokens)

        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(['<s_rvlcdip>'])[0]

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

    def load_model(self):

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image_path):
        image = Image.open(image_path)
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(self.task_prompt, add_special_tokens=False,
                                                     return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )
        # # turn into JSON
        seq = self.processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = self.processor.token2json(seq)

        # calcuate confidences
        gen_sequences = outputs.sequences[:, decoder_input_ids.shape[-1]:-1]
        probs = torch.stack(outputs.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        gen_probs = gen_probs.float()
        gen_log_probs = gen_probs.log()
        avg_log_prob_per_sequence = gen_log_probs.mean(-1)
        sequence_confidences = avg_log_prob_per_sequence.exp()

        # print("Generated Output:", seq['class'])
        # print('trubgv new\n', sequence_confidences)

        return seq['class'], sequence_confidences.tolist()[0]


root_dir = '/home/ankit/Desktop/Dataset/TNIAUSorted/'
model_path = "checkpoints/dnonut_classifcation_50.pth"
docclass = DonutClassification(model_path=model_path,
                               task_prompt="<s_rvlcdip>"
                               )


df = pd.DataFrame(columns=['filepath', 'cls', 'score'])
filedict = {}
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        cls = dirpath.replace(root_dir, '')
        file_path = os.path.join(dirpath, filename)
        filedict.update({file_path: cls})


for p in filedict.keys():
    if p.lower().endswith('jpg') or p.lower().endswith('jpeg') or p.lower().endswith('png'):
        cls, conf = docclass.predict(image_path=p)
        new_row = pd.Series({'filepath': p, 'cls': cls, 'score': conf})
        df = pd.concat([df, pd.DataFrame([new_row], columns=new_row.index)], ignore_index=True)
        break
