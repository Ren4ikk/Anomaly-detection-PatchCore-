poetry run python run.py `
  --train_dir mvtec_Defect_detection_dataset/capsule/train/good `
  --test_dir  mvtec_Defect_detection_dataset/capsule/test `
  --mask_dir  mvtec_Defect_detection_dataset/capsule/ground_truth `
  --save_path ./models/capsule.pt