$TrainDir = "mvtec_Defect_detection_dataset/capsule/train/good"
$TestDir  = "mvtec_Defect_detection_dataset/capsule/test"
$MaskDir  = "mvtec_Defect_detection_dataset/capsule/ground_truth"
$SavePath = "./models/capsule.pt"
$CoresetRatio = "0.1"
$BatchSize = "32"
$NumWorkers = "1"

poetry run python run.py --train_dir $TrainDir --test_dir $TestDir --mask_dir $MaskDir --save_path $SavePath --coreset_ratio $CoresetRatio --batch_size $BatchSize --num_workers $NumWorkers --verbose