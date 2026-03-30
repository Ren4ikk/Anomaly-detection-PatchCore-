$ModelPath  = "./models/capsule.pt"
$ImagePath  = "./mvtec_Defect_detection_dataset/capsule/test/crack/010.png"
$OutputDir  = "./results"
$Threshold  = "0.86"

poetry run python infer.py --model $ModelPath --image $ImagePath --output_dir $OutputDir