# Repository Submission — 提交内容

## 操作步骤

1. 打开: https://github.com/EBiM-Benchmark/submissions/issues/new/choose
2. 选择 **"Repository Submission"** 表单（不是 Technical Report）
3. 复制下面每个字段的内容到对应输入框
4. 勾选 Acknowledgement 复选框
5. 点击 "Submit new issue"

---

## 字段 1: Team name

```
world model
```

## 字段 2: Point-of-contact email

```
1373851641@qq.com
```

## 字段 3: Task

选择: **Task 3 — Assisted Living & Feeding**

## 字段 4: Public GitHub repository URL

```
https://github.com/CHANGCHENGGAO/ebim-world-model
```

## 字段 5: Did this submission use the simulator's ground-truth object poses?

选择: **No — we do not use the simulator's ground-truth object poses**

(注：策略在 real 模式下通过外部 3-D 感知话题获取物体坐标，fail-closed；不读取 USD/PhysX 场景真值，也不回退到固定坐标)

## 字段 6: Submission requirements

三个复选框全部勾选:
- ☑ The repository is public.
- ☑ It contains a Dockerfile encapsulating our work.
- ☑ It contains a README explaining how to run it.

## 字段 7: Optional supplementary links

```
Technical Report: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/Technical_Report_World_model.md
Policy module design: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/docs/policy_modules_design.md
Judge evaluation: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/docs/judge_evaluation.html
```

## 字段 8: Notes (optional)

```
Deliberately reduced, reliable strategy: the policy skips Stage 1 (table setup)
and Stage 2 (feeding), then runs Stage 3 (drop the whole bowl, beans included,
into the recycling bin) and Stage 4 (place the cup into the sink). We claim no
score; final scoring uses the organizer's own evaluation.

Key features:
• ISO/TS 15066 force monitoring — 20 N hard stop, contact detection at 5 N
• Closed-loop navigation — odometry + LaserScan with 3-attempt correction loop
• Fail-closed perception — external 3-D object poses, no simulator ground truth
• Spine height control — 0.80 m for door transit, 0.60 m for Stage 3/4 work
• Dual-LiDAR safety with autonomous recentre-and-retry recovery

This submission supersedes our earlier Technical Report (Issue #22). The policy
runs autonomously in real mode with no keyboard, GELLO, pedal or operator input.
```

## 字段 9: Acknowledgement

勾选:
- ☑ I understand source code is not required, but a Dockerfile and a run README are.
