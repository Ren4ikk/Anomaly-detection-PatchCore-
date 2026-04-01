$ModelPath  = "./models/capsule.pt"
$ImagePath  = "./mvtec_Defect_detection_dataset/capsule/train/good/020.png"
$OutputDir  = "./results"
$Threshold  = "0.86"

poetry run python scripts/infer.py --model $ModelPath --image $ImagePath --output_dir $OutputDir