$ModelPath  = "./models/capsule.pt"
$ImagePath  = "./mvtec_Defect_detection_dataset/capsule/test/faulty_imprint/014.png"
$OutputDir  = "./results"
$Threshold  = "0.62"

poetry run python infer.py --model $ModelPath --image $ImagePath --output_dir $OutputDir --threshold $Threshold