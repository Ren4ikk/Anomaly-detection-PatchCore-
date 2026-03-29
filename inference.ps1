$ModelPath  = "./models/capsule.pt"
$ImagePath  = "./mvtec_Defect_detection_dataset/capsule/test/poke/000.png"
$OutputDir  = "./results"
$Threshold  = "0.86"

poetry run python infer.py --model $ModelPath --image $ImagePath --output_dir $OutputDir --threshold $Threshold