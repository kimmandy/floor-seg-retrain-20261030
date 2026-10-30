import json,sys
from pathlib import Path
p=json.loads((Path(__file__).resolve().parent/'paths.json').read_text());run=Path(p['run'])
if '--test' in sys.argv:
    path=run/'test/metrics.json'
    if not path.exists():print('최종 test는 학습 완료 후 best에 대해 한 번 평가합니다.');raise SystemExit(0)
    data=json.loads(path.read_text());epoch=data['epoch']
else:
    path=run/'current/epochs.jsonl'
    if not path.exists():print('아직 완료된 epoch 검증 결과가 없습니다.');raise SystemExit(0)
    row=json.loads(path.read_text().splitlines()[-1]);data=row['valid'];epoch=row['epoch']
print(json.dumps(dict(epoch=epoch,split=data['split'],independent_evaluation=False,physical_scenes_shared=True,prior_model_exposure=data.get('prior_model_exposure'),evaluation_limit=data.get('evaluation_limit'),macro_f1=data['macro_f1'],mask_mAP=data['mask_mAP'],mask_AP50=data['mask_AP50'],classes=data['classes'],negative_images=data['negative_images'],hazard_fp_per_negative_image=data['hazard_fp_per_negative_image']),ensure_ascii=False,indent=2))
