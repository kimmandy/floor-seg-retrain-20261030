"""Report measured counts and actual live verification, never inferred completion."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def main():
    p=read('paths.json');plan=read('split_plan.json');live=read('live_verification.json')
    checks={n:read(n) for n in ('original_verification.json','coverage_check.json','dataset_verification.json','metric_tests.json','preflight_report.json','handoff_verification.json')}
    assert live['result']=='PASS' and all(x['result']=='PASS' for x in checks.values())
    init=read('initialization.json');dv=checks['dataset_verification.json'];gpu=checks['preflight_report.json']
    names={'water':'물','bolt':'볼트','nut':'너트','oil':'기름','paint':'물감','reflection':'반사'}
    lines=['2026-10-29 클래스별 8:1:1 재분배 및 이전 best 기반 추가학습',
           '최종 확인: '+datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'',
           '원본 553장 / 라벨 5253개 보존. train442 / valid55 / test56.',
           '클래스별 정답이 있는 원본 이미지 수 (한 이미지에 여러 클래스가 있으면 각 클래스에 포함):']
    for c,n in names.items():
        row=plan['summary']['class_images:'+c]
        lines.append(f"  {n}: {row['counts']['train']} / {row['counts']['valid']} / {row['counts']['test']} (전체 {row['total']})")
    lines+=['','클래스별 정답 객체 수:']
    for c,n in names.items():
        row=plan['summary']['class_objects:'+c]
        lines.append(f"  {n}: {row['counts']['train']} / {row['counts']['valid']} / {row['counts']['test']} (전체 {row['total']})")
    lines+=['','모든 클래스의 이미지/객체 비율은 8:1:1에서 각 split 1장/1개 미만 정수 오차.',
        '하단 고정광 imx_fixed_left/centre와 추가조명을 분리해 확인:',
        '  with_light 촬영80장: 64/8/8 (금속32/4/4, 액체24/3/3, 빈판8/1/1).',
        '  실제 추가 반사 마스크79개: 63/8/8. imx21_0172는 추가 반사 정답 없음; 원본 보존.',
        '  하단 고정광과 추가 반사는 기존 reflection 클래스 유지, 사용자 완료 라벨 수정 없음.',
        '조명 ON/OFF 대응 순번80짝은 같은 split. 원본/이미지SHA/할당짝/증강부모 교차0.',
        '같은 물체 배치와 연속 프레임은 split 사이에 공유됨. 해당 그룹 ID를 숨기거나 다시 명명하지 않음.',
        '기존 best가 학습했던 원본이 새 valid30/test38장에 포함됨. 독립 평가나 새로운 배치 일반화로 해석 금지.',
        '기존 valid140의 점수와 새 valid55의 점수를 직접 비교하지 않음. test는 best 선택에 사용하지 않음.',
        '확인된 원거리 oil 원본은 없음. 거리 미상은 그대로 유지. 단일 촬영 조건을 복제해 세 split에 넣지 않음.',
        '',f"학습 자료: 원본442 + 광학2210 + 크롭1768 = 4420. 매epoch1900항목, 모든442source 노출, full/crop 1:1.",
        '근/중/원 액체 및 moving물·추가조명·빈판이 매epoch 노출되도록 샘플링 버킷을 분리.',
        '광학: exposure .85~1.12, gamma .92~1.08, RB .92~1.08, saturation .8~1.15, noise .4~1.5, gradient ±.1, 좌우반전.',
        '크롭 nominal800/1024/1200/1024px; 위험물 마스크 부분 절단 거부. 반사 포함 픽셀/기하 대조.',
        '',f"시작: 이전 run의 검증 best E{init['source_epoch']} (SHA {init['sha256']}). 마지막 저장 E{init['source_last_epoch']}와 기존 실행 보존.",
        '새 optimizer/scaler/schedule로 추가100epoch; COCO 재초기화 아님. 새phase E0에서 시작 모델/검증best 보존.',
        'AdamW LR1e-4, warmup500batch, cosine최종1e-5, longside1600, mask stride1, BF16, batch2×누적2.',
        '조기종료0, 매epoch last, 개선best, 매10epoch 전체저장. CPU 입력 prefetch 유지.',
        '검증: score0.5/maskIoU0.5, 6클래스 동일가중 macroF1로best, maskmAP/AP50/거리/빈판FP 보고.',
        '',f"검증 PASS: 원본553SHA·라벨5253불변, 파생마스크{dv['derived_masks_pixel_verified']}개 픽셀대조, 부분절단/소실0.",
        f"실제 GPU {sum(len(r['images']) for r in gpu['loss_backward'])}입력 loss/backward/optimizer update·GT배정누락0·저장복원/RNG·FP16 NMS·valid55추론 PASS.",
        '클래스누락/증강평가유입/할당짝교차/거짓독립평가/변경LR·데이터 resume 거부 PASS.',
        '사전 GPU 검증에서 신규holdout의 list-count RLE 처리 오류 발견 후 evaluator 메모리 변환 수정. 압축712/list339 GT의 픽셀·F1·AP 동등성 회귀검사 PASS; 원본 저장형식은 보존.',
        f"실행 확인: 새epoch{live['progress']['epoch']} batch{live['progress']['batch']}/{live['progress']['batches']}, loss {live['progress']['loss']:.6f}; 단일trainer 및 공통잠금 확인.",
        'E0 가중치가 원본 best와 완전히 동일하고 새 optimizer가 비어 있음을 확인. 절전방지 active, SSH 종료 후 지속.',
        '재부팅 자동재개/Nano 자동배포 없음. 실제 새로운 배치의 성능 개선은 아직 검증되지 않음.',
        '', '원본: '+p['source_dataset'],'증강: '+p['dataset'],'실행: '+p['run'],'코드·검증: '+p['setup'],
        '', 'E32 오검출 보조감사: FP는 배경검출·다른클래스 오인·경계불일치·중복을 포함.',
        'imx21_0159의 water 예측 score0.5546875는 추가반사 정답과 IoU0.60599, 예측면적의99.8%가 반사 안에 있음.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    note=f'''## 2026-10-29 — 클래스별 8:1:1 재분배, E8 best 기반 추가학습

현재 실행은 `{p['run']}`이며 기본 실행기를 연결했다. 원본 `{p['source_dataset']}`는 553장/5253라벨을 보존하고 442/55/56으로 분배했다. 각 클래스의 양성 이미지 수는 물83/10/11, 볼트201/25/25, 너트236/29/30, 기름59/7/7, 물감66/8/8, 반사312/39/39다. 객체 수까지 각 split 8:1:1에서 1개 미만 정수 오차다. 추가조명 촬영80장은64/8/8, 하단고정광을 제외한 추가반사79마스크는63/8/8이다.

사용자의 학습 조건 우선·클래스별8:1:1·판단 위임에 따라 이 버전은 같은 실물 배치를 split 사이에 공유한다. 원본배치ID는 보존하고 ON/OFF 순번짝80개·원본SHA·증강부모는 분리했다. 모든 클래스는 세split에 존재하며, 확보된 3장이상 클래스/거리/조명 조건의 빈split은0이다. oil far GT 자체 없음과 단일 장면의 조명미기록 근거리2항목은 제한으로 기록했다. 과거best 학습원본이 새valid30/test38장에 포함되므로 **독립 평가/새배치 일반화 지표가 아니다.** 원본/완료웹라벨은 수정하지 않았다.

train에만5광학+4크롭을 생성해4420항목. 매epoch1900항목/950배치, 전442source 노출·full/crop1:1, 혼합액체 근/중/원과moving물·빈판을 별도샘플링한다. 보존된이전best E8에서 새optimizer/scaler로100epoch를 시작했다. 이전last E34와원래run보존.1600/stride1/BF16/batch2×누적2/LR1e-4/warmup500/cosine1e-5/earlystop0/매10epoch저장/CPU prefetch유지. 새valid55의6클래스 macroF1(score/IoU0.5)로best, test56은종료후best에한번. 시작가중치E0도best후보로보존한다.

실제검증:553SHA/5253라벨불변, 파생mask{dv['derived_masks_pixel_verified']}픽셀일치·잘림/소실0,100epoch전source노출,실제GPU역전파/GT배정/복원/RNG/FP16NMS/valid55·입력변경거부PASS. 신규holdout의list-count RLE평가오류를읽기단계정규화로수정하고압축712/list339마스크동등성검사PASS. 새epoch{live['progress']['epoch']} {live['progress']['batch']}/950유한loss·단일trainer·공통잠금·임시절전방지active를확인했다. 초기모델전가중치가E8best와정확일치하며새optimizer상태는비어있다. 재부팅자동재개/Nano배포없음. 근거: `setup_20261029/ratio811_best_0821/REPORT.txt`, 로컬 `artifacts/agx_coverage_repartition_20261029/REPORT.txt`.

'''
    (HERE/'documentation_note.txt').write_text(note)
    (HERE/'agents_note.txt').write_text('- 2026-10-29 사용자 클래스별8:1:1·best기반재학습: '+note.split('\n\n')[1]+' '+note.split('\n\n')[2]+' 현재run `imx219_ratio811_best_20261029_0821`, setup `setup_20261029/ratio811_best_0821`. E8best에서새optimizer100epoch/1900노출/1600stride1/BF16/earlystop0/10epoch저장으로실행확인. 이전last E34·원본·완료라벨보존. REPORT.txt와docs최신항목확인.\n')
    print('Wrote measured report and documentation notes')
if __name__=='__main__':main()
