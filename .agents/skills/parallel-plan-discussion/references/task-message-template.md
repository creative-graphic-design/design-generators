# Task message template for council participants

Send one message per participant via agmsg (`send.sh <team> main impl-plan-<slug> "<message>"`). Fill every `<...>` placeholder; delete sections that don't apply. Keep the round protocols intact — they are what make the relay work.

```markdown
## 背景

<プロジェクトの 1〜2 文説明>。あなたの担当は **<target-slug>**(<関連 issue/doc へのポインタ>)の<調査/実装プラン作成>です。同時に他 <N-1> 件の担当エージェントも稼働しており、**エージェント間で議論して <アジェンダの要約: 例 インターフェースを統一>** することがこのタスクの核心です。

- 担当一覧: <slug をカンマ区切りで>(agmsg role 名は impl-plan-<slug>)
- 対象ソースは <main checkout の絶対パス> を読み取り専用で参照(あなたの worktree には submodule の中身がありません)
- リポジトリへの変更・commit・push・PR は禁止。書き込みは <shared-drafts-dir>/ 以下のみ

## Round 1: 調査とプラン草稿(まずこれを完了)

1. <最初に読むべき issue/doc>: `gh issue view <umbrella>` と `gh issue view <target-issue>` を読む
2. 対象を深掘り: <調査観点のリスト。例: モデルクラス、エントリポイント、config、checkpoint 形式、データの扱い(座標系・正規化・ラベル語彙)、決定性>
3. プラン草稿を <shared-drafts-dir>/<slug>.md に書く。含めるもの:
   - <成果物の必須セクション。例: ディレクトリ構成(ファイル単位・クラス名・関数シグネチャ)、変換方針、processor 設計、重み変換とパリティテスト計画、未確定事項>
4. **提案**をプランの `## Interface proposal` セクションに書き、その要約(15 行以内)を他の全担当へ agmsg で送る:
   `~/.agents/skills/agmsg/scripts/send.sh <team> impl-plan-<slug> <peer> "<要約>"`
   提案に必ず含める項目: <アジェンダ項目の列挙。例: (a) 共通出力スキーマ、(b) 生成 API の署名、(c) 共通ユーティリティ候補と置き場所、(d) 命名規則>
5. ここまで完了したら main へ agmsg で「Round 1 完了」と報告し、**ターンを終了する**(inbox に peer 提案が届いていれば読んでおく)

## Round 2: 議論と収束(次の nudge で開始)

1. inbox の peer 提案を全て読む: `~/.agents/skills/agmsg/scripts/inbox.sh <team> impl-plan-<slug>`
2. 相違点への同意/反論を、**関係する peer へ直接** agmsg で送って議論する(必要なら複数往復)。あなたの担当固有の制約で統一案に乗れない点は根拠を示す
3. 合意を踏まえて <shared-drafts-dir>/<slug>.md を更新(提案セクションを合意版に書き換え、担当固有の逸脱は `Deviation` として理由つきで明記)
4. main へ「Round 2 完了(合意点 / 残る相違点の要約つき)」を agmsg で報告して**ターンを終了する**

<chair にのみ追加:>
あなたは今回の議論の **chair(まとめ役)** です。<chair が先に読むべき追加コンテキスト>。Round 2 では全員の提案・合意を集約した統一仕様を <shared-drafts-dir>/unified-interface.md に書いてください。統一仕様には <統一仕様の必須内容。例: 合意事項、共通スキーマ定義、担当別 Deviation 一覧、比較表(Model | ... | ...)、未解決の論点> を必ず含めてください。

<担当固有の注記: 対象特有の論点・既知の事実・注意点>

## 完了条件

- Round 1: <shared-drafts-dir>/<slug>.md 草稿 + 全 peer への提案送付 + main への報告
- Round 2: 合意反映済みプラン + main への合意/相違サマリ報告。PR / CI は不要です

## 報告

agmsg で main 宛に報告してください。Round 2 の報告には合意点と残る相違点の要約を必ず含めてください。
```

## Why the template is shaped this way

- The explicit peer list + role names let participants message each other without asking the coordinator for addresses.
- "ターンを終了する" at the end of each round is load-bearing: the coordinator can only relay rounds by nudging idle panes.
- The ≤15-line proposal cap keeps each participant's inbox digestible (they receive N-1 of these).
- Requiring `Deviation` entries with reasons prevents false consensus — targets with real constraints record them instead of silently ignoring the agreement.
- The agenda items in step 4 are enumerated concretely so the chair can check the unified spec for completeness item by item.
