# This .sh file uses BackBone: ViTAEv2_S ; Weights: vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth 
# Experiment: try on different configs to see which configs give best result on trial image

# IC15-finetuned
python demo/demo.py --config-file configs/ViTAEv2_S/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml --input demo/trial_img --output demo/trial_img_output --opts MODEL.WEIGHTS weights/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth

# pretrain
python demo/demo.py --config-file configs/ViTAEv2_S/pretrain/150k_tt_mlt_13_15_textocr.yaml --input demo/trial_img --output demo/trial_img_output --opts MODEL.WEIGHTS weights/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth

# TotalText-finetuned
python demo/demo.py --config-file configs/ViTAEv2_S/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml --input demo/trial_img --output demo/trial_img_output --opts MODEL.WEIGHTS weights/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth
