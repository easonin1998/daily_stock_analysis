# 选股引擎

DSA 将选股能力作为主项目的一部分维护。实现参考 [AlphaSift](https://github.com/ZhuLinsen/alphasift) 提交 [`9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf`](https://github.com/ZhuLinsen/alphasift/commit/9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf)，并按 Apache License 2.0 修改和分发。衍生文件保留来源头，许可证位于 `src/services/screening/LICENSE`，第三方声明见根目录 `THIRD_PARTY_NOTICES.md`。

## 代码边界

- `src/services/screening/`：快照、日 K、策略加载、过滤、评分、风险、LLM 重排与热点实现。
- `src/services/screening/strategies/`：随 DSA 版本发布的策略 YAML。
- `src/services/screening/pipeline.py`：筛选流程的直接入口。
- `src/services/screening/strategy_backtest.py`：基于本地日线缓存的 Martin 点时信号与组合回测，不替代历史 AI 报告验证。
- `src/services/screening_service.py`：DSA 业务编排，直接调用 pipeline，负责配置、数据源上下文、响应归一化、缓存与错误映射。
- `src/storage.py`：使用 DSA 现有 SQLAlchemy/SQLite 基础设施持久化已完成的选股运行，不另建文件数据库。
- `api/v1/endpoints/screening.py`：`/api/v1/screening` API。
- `apps/dsa-web/src/api/screening.ts` 与 `StockScreeningPage.tsx`：Web 调用与展示。

服务层静态调用 `screening.pipeline`、`screening.strategy` 和 `screening.hotspot`。核心逻辑不通过模块名探测、动态适配器或多套路由分发，因此代码结构、错误边界和打包收集目标均由主项目直接定义。

## 配置

默认关闭：

```dotenv
SCREENING_ENABLED=false
```

Web“基础设置”页展示“选股”开关。开启后左侧显示“选股”入口并允许执行策略、热点和选股任务；关闭后入口隐藏，选股 API 继续拒绝业务请求。

## Web 交互与飞书通知

Web 是选股的主交互入口。用户选择市场、策略和返回数量后提交后台任务，页面会持续显示进度；默认勾选“完成后推送飞书”，任务成功后发送专用结果卡片，列出市场状态、目标总仓位、候选数量、前列标的和条件单摘要。推送失败不会回滚或覆盖已经完成并持久化的选股结果。

飞书卡片中的详情链接使用浏览器提交任务时的网页来源，并附带 `runId`。打开 `/screening?runId=<run_id>` 会直接读取对应的历史结果，而不是重新执行筛选。为避免向手机发送无法使用或不安全的链接，只有 HTTPS 公网来源才会显示详情链接；`127.0.0.1`、`localhost`、局域网 IP 和普通 HTTP 来源仍会收到完整候选卡片，但不附网页链接。远程访问必须先启用管理员登录保护，再通过受保护的 HTTPS 域名或隧道接入。

常用可选项：

```dotenv
SCREENING_DATA_DIR=data/screening
SCREENING_SNAPSHOT_CACHE_TTL_SEC=300
SCREENING_SOURCE_CALL_TIMEOUT_SEC=
SCREENING_HOTSPOT_CALL_TIMEOUT_SEC=8
SCREENING_HOTSPOT_SEARCH_TIMEOUT_SEC=12
SCREENING_SNAPSHOT_CALL_TIMEOUT_SEC=60
SCREENING_DAILY_CALL_TIMEOUT_SEC=20
SCREENING_EASTMONEY_MIN_INTERVAL_SEC=1.0
SCREENING_EASTMONEY_JITTER_SEC=0.3
SCREENING_DAILY_FEATURE_LIBRARY_ENABLED=true
SCREENING_DAILY_FEATURE_LIBRARY_PATH=data/screening/daily_feature_library.json
SCREENING_DAILY_FEATURE_LIBRARY_MAX_AGE_DAYS=7
SCREENING_MARKET_REGIME_ENABLED=true
SCREENING_MARKET_REGIME_MIN_UNIVERSE_SIZE=100
SCREENING_MARKET_REGIME_UPGRADE_CONFIRMATION_DAYS=2
DAILY_LOOKBACK_DAYS=160
DAILY_ENRICH_MAX_CANDIDATES=120
DAILY_FETCH_MAX_WORKERS=4
INDUSTRY_PROVIDER=free_cached
SCREENING_SECTOR_STRENGTH_HISTORY_PATH=data/screening/sector_strength_history.sqlite3
SCREENING_SECTOR_FUND_FLOW_ENABLED=false
SCREENING_SEED_POOL_CORE_RATIO=0.55
SCREENING_SEED_POOL_STRENGTH_RATIO=0.20
SCREENING_SEED_POOL_ACTIVITY_RATIO=0.15
SCREENING_SEED_POOL_MAX_SECTOR_RATIO=0.20
```

路径、缓存、超时和限流项只影响选股链路。`SCREENING_SNAPSHOT_CACHE_TTL_SEC` 默认 300 秒，设为 `0` 可关闭新鲜快照复用。`SCREENING_HOTSPOT_CALL_TIMEOUT_SEC` 的正数值是默认热点 provider 单次板块、成分股或直接详情 fallback 调用的总预算，后续 fallback 和并行成分股源共用同一截止时间，并把剩余时间传入可终止的 AkShare 子进程与 HTTP socket；设为 `0/off/disabled` 只关闭这层总预算，各真实数据源仍保留自身硬超时。`SCREENING_HOTSPOT_SEARCH_TIMEOUT_SEC` 是用户主动新闻搜索的端到端截止时间，缓存 owner 等待、重新竞争和 provider 子进程共享同一个绝对 deadline；`0/off/disabled` 会回退到安全默认值 12 秒，而不是无限等待。完整示例以 `.env.example` 为准。

`martin_momentum_pullback` 至少需要 120 个有效交易日，默认抓取 160 个交易日。`DAILY_ENRICH_MAX_CANDIDATES` 同时是种子池上限：引擎不会再只按成交额截取前 N，而是从流动性核心、中期动量优先的强度池、交易活跃度和每日确定性轮换覆盖四条通道组池；有行业字段时，单一行业默认不超过种子池 20%，不足时再回填以保证样本数。本地指标库已有覆盖时，强度池优先使用真实 20/60 日涨幅和 EMA 趋势；未覆盖股票仍可通过轮换通道进入，避免一次建库失败后永久失去机会。增加该值会扩大相对强度比较池，同时增加首次运行的网络请求；后续运行优先复用 24 小时日线缓存。默认不启用会改写最终分数的 `POST_ANALYZERS`；需要 DSA 或外部深度分析时再显式开启。

## 当前内置策略

内置策略只保留 `martin_momentum_pullback`（Martin 动量回调），适配 A 股 T+1、涨跌停和流动性约束。原有双低、价值、低波、超跌反转、资金热度、放量突破等策略文件已移除，不再出现在策略列表中，也不能发起新的运行；数据库中已有的旧策略结果仍可按 `run_id` 读取。

硬过滤先要求全市场上涨家数占比不少于 35%、涨跌幅中位数不低于 -1.5%，避免在明显 risk-off 环境逆势开新仓；个股需非 ST、股价不低于 1.5 元、当日成交额和 20 日平均成交额均不少于 1 亿元、市值不少于 30 亿元、至少 120 个日线样本，并排除一字涨停。1.5 元是避免过度接近 1 元交易类退市风险区的执行护栏，不把 4 元误当监管边界；1.5-2 元与 2-4 元股票仍可入围，但在风险层分别施加较强与轻度软惩罚。随后共同要求 ADR20 位于 3.5%-8%、20 日相对强度位于候选池前 30%、60 日相对强度位于前 5%、EMA9 > EMA21 > EMA50 且 EMA21/EMA50 上行、距 120 日高点不超过 15%。3.0 起，回调深度、缩量缩波、支撑共振和触发枢轴不再作为所有形态共用的单一路径，而由平台突破、首次回踩、紧缩延续三类形态分别判定。60 日前 5% 仍是领涨股硬门槛；触发日收盘在全天区间的位置和相对前一日量能主要用于形态确认与质量评分。

固定七维评分如下，权重合计 100%：

| 维度 | 权重 | 主要证据 |
| --- | ---: | --- |
| 趋势质量 | 20% | EMA 多头、EMA21/50 斜率、距 120 日高点 |
| 相对强度 | 20% | 20/60 日收益在日线候选池内的百分位 |
| 板块共振 | 15% | 点时行业强度、趋势宽度、领涨股及高 ADR 领涨股数量；缺失时明确标记 unavailable，并仅在评分层使用兼容中性值 |
| 缩量收敛 | 15% | 近 3 日波幅/成交量相对前序窗口的收缩与回调深度 |
| 支撑共振 | 15% | 价格与 EMA9、EMA21、近 60 日高点锚定 VWAP 的接近程度 |
| 触发质量 | 10% | 收盘相对前一交易日高点的位置、全天区间收盘位置、相对前一日量能，并惩罚单日追涨 |
| 流动性 | 5% | 20 日平均成交额优先，缺失时回退快照成交额，并参考换手率 |

ADR20 定义为最近 20 个交易日 `mean((high / low - 1) * 100)`，衡量股票日内可交易波动，不是美国存托凭证，也不等同于包含跳空的 ATR。

每个最终候选都会附带结构化 `trade_plan`。形态尚未完成自身触发条件时，状态为 `watch_wait_trigger`；形态已经确认但不在 `tradable_setup_types` 交易许可清单中时，状态为 `watch_only_setup`。两者均明确标记“不下单”。只有具备交易许可的确认形态才会进入 `buy_signal`，同时给出下一开盘集合竞价的最低/最高可接受价、初始防守价、1R 保本观察价、2R 观察价和 10 日跟随要求。3.3 当前继续识别平台突破、首次回踩、紧缩延续三类形态，但只允许 `first_pullback` 进入交易；另外两类继续进入观察统计和飞书展示，便于积累样本，不会生成可武装的条件单。冻结的 3.0 profile 仍保留三类形态全部可交易，供历史复现和消融对照。正式策略不再设置固定最长持有期，趋势和移动防线未破坏时允许继续持有。开盘区间不再固定为对称 ±2%：波动基准取信号日 `ATR20%` 与 `ADR20%` 的均值（缺一则使用另一个），低开容忍取基准的 30% 并限制在 1%～1.5%，高开容忍取基准的 50% 并限制在 1.5%～2%；高波动只会让区间逐步接近上限，不会把高开追价放宽到 2% 以上。最低价还必须守住前一日高点和 EMA21，实际成交后仍需通过 3% 技术止损距离上限。防守价由前高、EMA21、锚定 VWAP、ADR 缓冲和最大初始风险共同约束，不由 LLM 临时发挥。信号日的 1R/2R 仅按参考收盘价预估，实际成交后由 `rebase_martin_trade_plan_after_fill` 按成交价和初始防守价重算；2R 是移动防守与减仓评估参考，不是未经验证的固定止盈点。仓位不再按候选数量机械等权：默认以账户权益的 0.5% 作为单笔风险预算、2.5% 作为全部持仓技术风险热度上限，并只用实际成交价到初始防守价的技术止损距离反推股数；单只名义仓位不得超过 20%，技术止损距离超过 3% 时放弃交易，A 股数量向下取整到 100 股后仓位不足账户权益 12% 时也不下单。`1.0 × ADR20` 独立作为集合竞价跳空压力指标展示，不参与计仓。未提供账户权益时只输出权重上限、最低有效仓位、风险口径和公式，不伪造股数。

2.8 起，策略在单日宽度硬过滤之外增加多日市场状态层。状态分数由全市场趋势（25 分）、五日涨跌宽度（30 分）、领涨股数量与表现（30 分）、极端下跌冲击（15 分）组成，并持久化最近的盘后横截面；单一交易日的风险恶化立即降级，改善则需连续两个交易日确认，避免在震荡期频繁切换。3.0 继续把市场状态作为组合硬上限，并在风险恢复时渐进开放：初始最多开放两个仓位，已有持仓达到 1R 后增加一个名额，初始止损失败则收回一个名额；`risk_off` 后恢复从一个名额重新开始。市场状态默认只约束新开仓和仓位上限，不强制卖出已有持仓。

| 市场状态 | 目标总仓位 | 最多持仓 | 单笔风险 | 组合热度 | 单只上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 强势进攻 `strong_risk_on` | 100% | 5 | 0.50% | 2.50% | 20% |
| 正常进攻 `risk_on` | 80% | 4 | 0.40% | 2.00% | 20% |
| 中性防守 `neutral` | 40% | 2 | 0.25% | 1.00% | 15% |
| 风险关闭 `risk_off` | 0% | 0 | 0% | 0% | 0% |

中性和正常进攻状态会同步降低最低有效仓位门槛，避免风险预算已缩小却被固定 12% 门槛全部拒绝；`risk_off` 下候选仍会出现在观察结果中，但确认信号标记为 `market_blocked`，不会生成买入条件单。全市场技术特征少于 `SCREENING_MARKET_REGIME_MIN_UNIVERSE_SIZE` 时状态为 `unavailable`，保持原有基础仓位规则，不因数据缺失误判为风险关闭。可用 `SCREENING_MARKET_REGIME_ENABLED=false` 或回测参数 `--no-market-regime` 做对照研究。

持仓期间每天收盘后更新一次有效防守价。3.3 使用 `tranche_cost_profit_floor_volatility`：每次初始建仓或加仓都保存独立的费用后单位成本 `Cᵢ`、当时保护价 `Sᵢ`、单位风险 `rᵢ = Cᵢ - Sᵢ` 和剩余股数 `qᵢ`。组合当前R定义为 `Σ[qᵢ × (收盘价 - Cᵢ)] / Σ(qᵢ × rᵢ)`；达到1R、2R后的加仓、3R/5R成熟阶段、EMA9保护和5R衰竭减仓统一使用该组合R，不再使用首笔买入的R冒充整个仓位的R。每次加仓后组合构成发生变化，开盘成交只更新批次账本，必须等加仓日首个有效收盘后才以新组合的收盘R重建峰值R，避免集合竞价成交价提前激活退出阶段。

3.3 的组合回撤高水位不再独立沿用加仓前最高收盘价，而是始终由同一时点的组合峰值R反推：`高水位价格 = [Σ(qᵢ × Cᵢ) + 峰值R × Σ(qᵢ × rᵢ)] / Σqᵢ`。因此加仓后的平均成本、风险分母、峰值R和回撤锚点使用同一批剩余仓位；回测逐笔结果同时输出 `position_high_water_price` 供审计。

3.3 的组合利润底线按 `Σ(qᵢ × Cᵢ) + 目标R × Σ(qᵢ × rᵢ)` 计算，再除以剩余股数和预计卖出净保留率；因此1R/2.5R利润底线覆盖全部剩余仓位、买入费用、预计卖出滑点、佣金和印花税。1R～3R只启用 `clamp(1.0 × ADR20, 5%, 8%)` 波动回撤，不设置0R保本底线；组合峰值达到3R后才启用1R利润底线并按 `clamp(1.5 × ADR20, 7%, 12%)` 管理，达到5R后将底线抬至2.5R并按 `clamp(1.75 × ADR20, 9%, 15%)` 管理。所有退出继续由收盘确认并在下一交易日开盘集合竞价执行，不保证跨夜跳空后仍能以利润底线成交。冻结的3.1保留旧首笔R自适应模型，冻结的3.2关闭自适应退出并使用固定8%回撤，二者只用于复现和消融对照。

## 本地日线指标库

本地指标库把已完成日 K 计算得到的 20/60 日涨幅、ADR20、EMA 趋势、回调深度、量价收敛、支撑与触发位置等特征按股票代码持久化。正常筛选先把有效期内的指标合并到快照初筛结果，再建立种子池；20/60 日相对强度百分位按完整快照合格池计算，不在 120 只种子池内缩小比较基准。种子池中的股票仍会读取原始日线缓存或实时源复核，成功后回写指标库。这样全市场排序不必每次发起数千个日线请求，同时保留最终候选的新鲜度确认。

首次建库或补齐覆盖可运行：

```bash
# 默认只处理一批 200 只，可重复执行并断点续建
python scripts/build_screening_daily_library.py

# 一次持续处理当前符合快照基础门槛的全部股票
python scripts/build_screening_daily_library.py --batch-size 250 --workers 4 --all

# 回测研究专用：完整当前 A 股快照，不做当前成交额/市值/价格预筛
python scripts/build_screening_daily_library.py \
  --universe-mode full-a-share \
  --library-path data/screening/backtests/full_a_share_daily_feature_library.json \
  --batch-size 250 --workers 4 --all
```

脚本会先吸收已有 `daily_history` 缓存，再按确定性顺序请求尚未覆盖的股票。建库默认要求 `qfq` 前复权口径；腾讯、AkShare 和 Baostock 的前复权缓存可复用，Sina 不复权缓存及旧的未知口径缓存会被拒绝并重新抓取，禁止在同一横截面混算收益、均线和止损。仅当实时快照、交易日历和盘后/非交易日状态均可确认时，才会把口径兼容的最近完整快照补成已完成日 K；不能确认时宁可跳过，不伪造交易日。初始建库完成后，正常选股会在检测到指标库落后于最近完整交易日时执行相同的本地增量维护，只读取已有缓存和当日快照，不为全市场重新发起日线请求。中断不会破坏现有库，后续再次运行会跳过有效期内已完成的股票。指标库是派生索引，删除后可从日线缓存或数据源重建；原始日线缓存仍是可追溯依据。

`full-a-share` 指的是完整快照进入建库任务，不表示每只股票都具备足够历史。脚本输出的 `entry_count` 是已登记代码数；真正可参与 Martin 计算的覆盖还需要检查 `daily_data_points >= 120`、最新完成交易日和前复权口径。研究时生成到自定义 `--library-path` 的全A库不会自动替换日常运行库；完成验收后应将同一全A库继续维护在 `SCREENING_DAILY_FEATURE_LIBRARY_PATH` 指向的正式路径。

## 免费板块数据与每日采集

日常筛选推荐 `INDUSTRY_PROVIDER=free_cached`：Baostock 提供稳定行业归属，AkShare行业/概念热度只读取盘后任务已经生成的本地缓存，不在用户请求内批量访问网页。盘后脚本默认使用联网的 `free` 组合刷新这些缓存。板块强度的主体仍由本地全A前复权日线计算，包括20/60日相对强度、EMA趋势宽度、领涨股与高ADR领涨股数量；网页热度和可选资金流只占外部增强部分，失败时不会阻断本地计算。

完成全市场日线更新后运行：

```bash
python scripts/collect_screening_sector_data.py

# 免费资金流接口可能较慢，建议只在独立盘后任务中显式开启
python scripts/collect_screening_sector_data.py --with-fund-flow
```

采集器要求同一完成交易日的日线覆盖不低于80%，低于门槛会拒绝写入，避免用残缺股票池制造虚假的板块宽度。行业宽度和领涨占比使用5个成员的全市场先验做小样本收缩，少于5个有效成员的行业不能获得领涨优先资格，避免“单只股票上涨即100%板块宽度”。行业日度指标与成分有效期分别保存到 `sector_strength_history.sqlite3`；行业变更只新增一条有效期记录，不再每天为约五千只股票重复保存相同板块指标。回测按信号日精确连接当日行业指标和当时有效成分，首个已采集日期以前保持 `unavailable`，不会用今天的概念成分倒填历史。概念成员变化快，当前只用于采集日起的实盘增强；在积累逐日快照前不进入正式历史回测。

## Martin 日线组合回测

策略回测与现有 `BacktestService` 职责不同：`BacktestService` 验证已经保存的 AI 分析记录，本模块则把 Martin 硬过滤和七因子排序还原到历史日线横截面，并模拟一个可交易组合。二者不会共用结果表，也不会互相覆盖。

默认运行示例：

```bash
python scripts/run_martin_strategy_backtest.py \
  --start-date 2026-08-01 \
  --end-date 2026-10-16

# 对绝对盈亏最大的 5 个成交标的追加 AkShare 主取 + Baostock 复核
python scripts/run_martin_strategy_backtest.py \
  --start-date 2026-08-01 \
  --end-date 2026-10-16 \
  --validate-samples 5

# 使用独立的完整当前 A 股日线库回测
python scripts/run_martin_strategy_backtest.py \
  --start-date 2026-04-01 \
  --end-date 2026-10-17 \
  --initial-cash 50000 \
  --universe-mode full-a-share \
  --library-path data/screening/backtests/full_a_share_daily_feature_library.json

```

回测会按“120 根策略预热 + 回测区间预计交易日 + 15 日缓冲”自动提高所需日线窗口，并可复用更长窗口的兼容缓存。由于数据源可能返回少于请求窗口的记录，长区间回测还会校验缓存的实际起止日期；当声明窗口更长的缓存反而缺少区间起点历史时，会在同一股票的兼容缓存中选择实际预热更完整且覆盖区间结束日的版本。较短窗口键的复用数量会在数据诊断中单独披露。报告同时展示区间起点具备完整 120 根历史的覆盖率；上市时间较晚的股票仍可在积累足够历史后参与，但不会计入期初完整预热。覆盖率低于 90% 时会写入限制项，不能把早期缺失信号解读为策略主动空仓。两个月区间建议先建立至少 220 根日线缓存：

```bash
python scripts/build_screening_daily_library.py \
  --lookback-days 220 \
  --max-age-days 0 \
  --batch-size 250 \
  --workers 4 \
  --all
```

回测只保留一套 Martin 选股与持仓系统，不再把“左侧/右侧”作为两个策略。次日开盘低开与高开边界均按 ATR20/ADR20 自适应，高开上限不超过 2%。3.0 先用动量、相对强度、EMA 趋势和 ADR 建立共同候选，再分别识别平台突破、首次回踩、紧缩延续；每类形态使用自己的历史枢轴价做收盘确认，不设统一突破幅度上限。这里的三类日线规则是 DSA 对 Martin“领先股、紧缩、突破/回踩后重新转强”思想的可回测适配，不宣称为其原始逐字硬条件。防追高继续由信号日涨幅、一字板过滤、触发质量、次日自适应开盘区间和成交后技术止损共同承担。信号只读取当日收盘及以前的数据，统一在下一交易日开盘集合竞价成交。全程不使用分时、盘中高低点止损或同日反向成交。

组合基础上限为最多 5 只、100 股整数手，实际开仓上限再按多日市场状态和渐进开放名额共同缩放。3.0 不再固定冷却 3 个交易日：退出当日禁止反手，此后只有出现新的收盘形态确认才允许重新入场。单股先建约 10% 试仓；3.3 只在整个剩余仓位达到1R、2R后各增加约5%，每次加仓后重新计算费用后加权成本和组合R，不向亏损仓位摊低成本，最终仍受20%单股上限约束。每笔基础风险预算为账户权益 0.5%，组合技术风险热度不超过 2.5%；实际成交后技术止损距离超过 3% 时放弃交易。`1.0 × ADR20` 仍作为独立跳空压力场景记录，不直接压缩股数。默认不设固定持仓天数上限；`--max-holding-days` 只保留为对照研究参数。这里的“集合竞价成交”由日线开盘价近似，不表示能够还原委托、排队、撤单和真实成交概率。

`src/services/screening/strategies/archive/` 保存冻结的2.8、3.0、3.1、3.2参数；`--strategy-profile 2.8` 至 `3.2` 可复现旧执行方式，默认 `--strategy-profile 3.3`。3.3 的多形态、板块、渐进仓位和盈利加仓可分别用 `--disable-multi-setup`、`--disable-sector-leadership`、`--disable-progressive-exposure`、`--disable-pyramiding` 关闭；`--disable-adaptive-exit` 可直接与冻结3.2对照。可用 `--ablation-label` 区分产物，回测文件名包含策略版本，各版本不会互相覆盖。

正式多年度消融可运行 `python scripts/run_martin_ablation_suite.py --start-date 2024-01-01 --end-date 2026-10-16`。该入口只加载一次全市场点时面板，再依次执行冻结v2.8、v3.0、旧自适应v3.1、固定退出v3.2、当前v3.3及其他模块对照，并输出统一的Markdown、JSON、CSV汇总；各组交易明细与资金曲线仍独立保存。模块贡献按“当前v3.3指标减去对照组指标”计算。若没有同区间点时板块历史，板块组只能验证 unavailable 降级契约，不应被解释为板块因子无效。

默认成本为买卖佣金各 3bp、买卖滑点各 5bp、卖出印花税 5bp。佣金按券商综合佣金处理，不重复叠加交易所经手费；不模拟单笔最低 5 元佣金。开盘接近对应板块涨跌停幅度时不假设可以买入或卖出，期末未平仓头寸按最后收盘价扣除预计卖出成本。自适应开盘区间的比例和上下限均可通过脚本参数覆盖；`--entry-gap-min-pct` 与 `--entry-gap-max-pct` 仅用于固定阈值对照实验，默认不传。报告会完整记录实际参数，每笔成交也保存当时的波动基准和允许区间。

产物写入 `data/screening/backtests/`：Markdown 用于阅读，JSON 保存完整规则、数据诊断、信号统计、净值与逐笔成交，CSV 分别保存单策略净值和交易。目录位于 `data/` 下，不作为代码提交。报告必须同时阅读以下限制：

手机端可将最新回测结果直接推送为飞书 JSON 2.0 卡片：

```bash
python scripts/push_martin_backtest_feishu.py
```

也可通过 `--report-json` 指定某个回测 JSON。默认依次发送总览、月度资金曲线与退出归因、分页逐笔交易卡片；逐笔页明确展示股数、买卖成交价、买入总支出、卖出净回款、净盈亏与退出原因，其中总支出包含买入佣金，净回款已扣除卖出佣金和印花税。`--summary-only` 可只发总览。完整 JSON/Markdown/CSV 仍保留在本地产物中供复核。

- 默认股票池由当前快照基础门槛建成，存在幸存者偏差和当前流动性偏差。`full-a-share` 模式覆盖当前完整 A 股快照并取消当前成交额、市值和价格预筛，但历史已退市股票仍不在当前快照中；历史 ST、总市值、退市与停复牌状态尚未点时重建，当前 ST 名称会在完整市场模式下保守排除。
- 板块库按 `日期+股票代码` 保存当日行业归属和行业强度，不跨日填充。历史不足时对应记录为 unavailable；评分层可使用兼容中性分，但报告与交易上下文必须保留缺失状态，不能把它解释为板块已确认。
- 回测默认只接受明确标记为 `qfq` 的腾讯、AkShare、Baostock、Tushare 或 DSA 降级链缓存；Sina 不复权数据会被排除。`--validate-samples` 对成交影响最大的股票执行 AkShare/Baostock 双源复核，按归一化价格路径和日收益差异判定，避免仅因固定复权缩放差异误报；它仍是抽查，不等同于全池双源认证。
- 两个月级别样本只能用于验证实现和观察分布，不能证明策略长期有效；应继续扩展点时股票池、历史窗口和滚动样本外测试。

## API 契约

| 路径 | 方法 | 行为 |
| --- | --- | --- |
| `/api/v1/screening/status` | GET | 返回开关、引擎状态、契约版本、参考项目和数据源健康信息 |
| `/api/v1/screening/strategies` | GET | 返回选股策略 |
| `/api/v1/screening/hotspots` | GET | 读取缓存或显式刷新热点题材 |
| `/api/v1/screening/hotspots/{topic}` | GET | 返回题材路线、成分股与核心股；`include_search=true` 时按需搜索近期消息 |
| `/api/v1/screening/screen` | POST | 同步执行选股；可传匿名 `variant_seed` 生成有界的近分候选组合，并以 `notify=true` 请求飞书完成通知 |
| `/api/v1/screening/screen/tasks` | POST | 提交后台选股任务；请求字段与同步接口一致，可附 `web_base_url` 生成运行详情链接 |
| `/api/v1/screening/screen/tasks/{task_id}` | GET | 查询任务进度、错误或最终结果 |
| `/api/v1/screening/history` | GET | 按策略、市场查询最近完成的选股运行摘要 |
| `/api/v1/screening/history/{run_id}` | GET | 读取一条持久化的完整选股结果 |
| `/api/v1/screening/source-history` | GET | 汇总历史运行中的快照源命中、错误和降级次数 |

后台任务使用 `report_type=screening_screen`，Web 会保存活动任务 ID，并在页面恢复时继续轮询。任务状态会分别提示全市场快照、候选上下文、LLM 重排、最终评分和新闻事件增强等阶段；完成后的结果同时写入 DSA 数据库，因此服务重启后仍可按 `run_id` 查询。

## 核心流程

```text
策略加载
  -> 全市场快照与字段标准化
  -> 流动性/市值快照初筛
  -> 多通道种子池与行业配额
  -> 本地缓存/直连日 K 特征与跨股票相对强度
  -> Martin 硬过滤与七维评分
  -> 候选上下文补充
  -> LLM 重排（可降级）
  -> 风险、近分轮换与保质量行业配额
  -> Top 候选 DSA 行情/基本面/新闻增强及 AkShare/Baostock 终选复核
  -> API 归一化响应与 DSA 数据库持久化
  -> 用户按需进入 DSA 单股深度分析
```

- 全市场快照在短 TTL 内优先复用最近成功结果；新缓存会记录完整且有序的数据源优先级，只在当前优先级与写入时一致时复用，因此同一来源链中的后备源结果可以加速后续请求，修改来源配置后则会重新读取实时数据。缓存过期后按配置优先级逐源尝试，单一数据源失败后继续降级，并记录 source health 与 last-good 缓存。当前 Sina、Efinance、AkShare/东财和 Tushare 快照接口均不提供增量游标或变更序列，因此 TTL 内可以零请求复用，TTL 到期后仍需重新读取全表；本地比较前后差异不能减少上游传输量，不作为“增量拉取”宣传。
- 有 `TUSHARE_TOKEN` 时默认优先 Tushare，否则默认从 Sina 开始；显式 `SNAPSHOT_SOURCE_PRIORITY` 始终优先。
- 种子池日 K 优先读取选股缓存并使用腾讯、Sina、AkShare、Baostock 等本地直连降级链；仅在这些来源失败时进入较重的 DSA `DataFetcherManager`。该桥接是请求级闭包，不替换进程级函数，因此重叠请求之间不会共享 wrapper。腾讯日线缺少成交额时，按其“手”口径使用 `成交量 × 收盘价 × 100` 估算；Sina 按其“股”口径使用 `成交量 × 收盘价`。估算值只参与流动性门槛，不改变价格或量价形态因子，历史缓存会在计算前应用相同口径。
- 最终组合先执行软集中度惩罚，再应用每板块最多 2 只的保质量配额；只有跨板块备选与被替代候选分差不超过 3 分时才替换，否则保留更强候选并记录原因。配置 `CN_QUANT_DATA_MODE=akshare_baostock` 时，最终候选额外运行 AkShare 主取、Baostock 前复权校验，并把校验状态写入 `dsa_context.daily_validation`；校验失败按 fail-open 处理并明确记录。
- 候选新闻优先复用 DSA 搜索服务；主搜索不可用或无结果时，降级到 AkShare `stock_news_em` 国内资讯，并硬性剔除缺少发布时间、超过 48 小时或明显未来时间的数据。
- LLM 重排前只补充有限候选上下文，最终候选再补行情、基本面、新闻和摘要，控制请求量。
- 默认最终本地分数直接采用 Martin 七维评分，不再由旧 `scorecard` 二次改写。显式配置多个后置分析器时，每一步完成后仍按最新分数重排，因此后续 `dsa` 与 `external_http` 的 `POST_ANALYSIS_MAX_PICKS` 上限作用于当前真实前列。远程状态按实际提交候选记录，外部响应中的超限代码不会改写未提交候选；启用远程分析时轮换只会在已完成相同分析的候选之间发生。
- 模型、渠道、base URL、额外 headers、fallback、timeout 和 token 上限在单次调用范围内注入，不改写用户配置；主模型即使 HTTP 调用成功，但返回空内容、非 JSON 或覆盖率不足，也会继续尝试已配置的备用模型。最终 JSON 必须在 `content` 块或 `output` 块中；`reasoning_content`（链式思考）被视为内部辅助，不作为最终结果。
- 热点榜单刷新与选股长流程可并行执行；列表默认不批量预取详情，用户选中具体题材时才加载该题材详情；显式刷新后若继续保留当前题材，Web 会同步绕过详情缓存重拉该题材，保证榜单与详情来自同次刷新。
- 热点成分股并行获取东方财富与同花顺数据，并按固定数据源优先级合并，避免响应先后改变重复股票的字段：正数 `SCREENING_HOTSPOT_CALL_TIMEOUT_SEC` 作为默认 provider 整次调用的共享预算，板块列表、成分股以及引擎失败后的直接详情 fallback 都进入该预算；AkShare/东方财富的可终止子进程与同花顺 HTTP connect/read timeout 每一步只使用剩余时间，fallback 不会重新获得完整预算。直接详情 fallback 不再额外启动无法由该预算强制回收的实时行情预取，行情字段以已受控的成分股源结果为准。超时子进程会 terminate/kill 并回收；进程级并发槽限制活跃任务数量，方法返回前会等待已接纳的 worker 结束，不遗留后台线程。关闭整次调用预算时，各单源仍保留默认硬超时；单源失败不会阻止另一源和本地核心股回退。
- “搜索最新消息”复用 DSA 原生搜索服务的 provider 优先级、SearXNG 公共实例能力、结果缓存与请求合并，同一缓存键只有 owner 可以启动供应商链；owner 未产出可缓存结果时，等待者重新竞争且未获得所有权的请求继续等待，但缓存等待、抢占和 provider 执行共用请求的绝对 deadline，不会在排队后重新获得完整超时。搜索只补充有链接的事件/催化，不从网页内容推断板块成分股；增强记录分别追加到展示路线和原始时间线，不覆盖已有 `timeline`。搜索由用户主动触发，摘要在本地确定性压缩，不调用 LLM；响应以 `available`、`no_results`、`unavailable` 区分有结果、有效空结果和超时/容量/供应商失败，Web 不会再把运行失败提示成“没有近期消息”。搜索增强仅存在于本次响应，不写入或续期共享热点详情缓存，默认详情请求不会看到搜索状态或增加搜索等待。供应商子进程在启动、执行或清理任一阶段失败时都会释放进程级容量。
- 热点实时请求失败时优先使用 last-good cache；无缓存时返回稳定空态与明确错误。
- 盘前任务尚无当日完整 K 线时，以本地日线指标库的最近完整交易日作为分析日，并据此更新市场状态、读取或重建全市场行业强度；不会因为 7:30 尚未开盘而把市场状态和板块强度整块显示为缺失。
- “板块/题材热度”拆分为三个独立口径：本地量价热度来自全市场 20/60 日相对强度、趋势宽度与高 ADR 领涨股；外部题材异动来自免费热点榜并支持 last-good cache；资金流只在上游返回真实字段时展示。任一外部层失败只降级该层，不会抹掉本地行业强度，也不会用中性值伪装成实时数据。
- 飞书选股完成卡片固定展示市场状态、行业量价热度、题材异动、候选与条件单、数据质量；即使本次零候选，也会保留分析日期、全市场覆盖、板块排行及各外部层的可用状态。
- 飞书通知刷新题材时使用轻量榜单模式，只读取一次板块异动摘要并写入 last-good cache，不为 Top 题材逐一抓取成分股；网页题材详情仍按用户选择加载完整成分与事件路线。

## 结果轮换

Web 会在浏览器本地生成一个不含用户信息的匿名种子，并随同步或后台选股请求传入 `variant_seed`。如果 Web Storage 不可读写，则在当前页面会话的模块内存中复用同一个临时种子，保证同步和后台任务入口一致。服务端将匿名种子与本次运行 ID、市场和策略共同作为扰动输入：不同浏览器以及同一浏览器的不同运行，都可能在质量接近的候选中看到不同股票。

扰动不是随机改分，也不会绕过策略：硬过滤、风险否决、因子/LLM 得分、最终评分和组合集中度惩罚全部先执行。默认本地评分覆盖完整短名单；启用有数量上限的远程后置分析时，只有完成相同分析的候选才可参与轮换。分析器产出的候选顺序是轮换输入的权威顺序，并列分不会再按股票代码重新排序；原 Top-N 的前半部分和明显高于截止分的候选始终受保护，只有后半部分名额可从不低于原截止分 1.5 分的近分池中抽取，入选候选继续保持该输入相对顺序。种子不写入选股结果或运行历史。未传 `variant_seed` 或将轮换比例设为 0 时返回严格输入 Top-N，保持脚本与旧客户端兼容。

## 缓存与持久化

| 数据 | 位置 | 有效期/行为 |
| --- | --- | --- |
| 全市场快照 | `data/screening/snapshot.last_good.json` | 默认 5 分钟内直接复用且不标记 fallback；过期后请求实时源，实时源全部失败时仍可按最大陈旧时间约束回退并标记 stale/fallback |
| 多日市场状态 | `data/screening/market_regime_history.json` | 每个已确认收盘日保存一条全市场观察，用于状态评分和两日升级确认；盘前使用本地指标库最近完整交易日更新，属于可重建派生数据 |
| 板块强度历史 | `data/screening/sector_strength_history.sqlite3` | 标准化保存行业日度强度和股票行业有效期；回测按交易日及当时有效成分连接，不向过去回填未来快照 |
| 个股日 K | `data/screening/daily_history/` | 按代码、来源和回看窗口分键，默认 TTL 24 小时；实时源全部失败时可使用过期缓存并标记 stale |
| 日线指标库 | `data/screening/daily_feature_library.json` | 按代码保存日 K 派生指标；默认只将最近 7 天内的条目合并进种子池，支持脚本分批断点续建 |
| Martin 策略回测 | `data/screening/backtests/` | 按起止日期输出 Markdown、JSON、逐笔交易和每日净值；属于可重建研究产物，不写 DSA 业务数据库 |
| 行业/概念映射 | `data/screening/industry_provider_cache/` | Baostock 全市场行业映射或 AkShare 板块映射默认 TTL 24 小时，并保存可用的板块热度历史 |
| 热点列表与历史 | `data/screening/hotspots.json`、`hotspot.history.jsonl` | 显式刷新写入；实时失败时回退最近可用快照 |
| 热点详情 | `data/screening/hotspot_details/` | 默认 TTL 30 分钟；只缓存结构化基础详情，显式消息搜索不写入或续期该缓存；实时失败时可回退过期详情并返回陈旧时长 |
| DSA 实时行情 | `DataFetcherManager` 的行情缓存 | 默认 TTL 10 分钟，沿用 `REALTIME_CACHE_TTL` |
| DSA 基本面/资金流 | `DataFetcherManager` 的基本面缓存 | 默认 TTL 120 秒，沿用 `FUNDAMENTAL_CACHE_TTL_SECONDS` |
| DSA 新闻/公告事件 | `SearchService` 内存缓存 | 成功结果默认 TTL 10 分钟；同题材并发请求在父进程合并，实际供应商链在限流、可终止的子进程中执行；服务重启后重新查询 |
| 完整选股结果 | DSA 数据库 `screening_runs` 表 | 完成后按 `run_id` 幂等写入；数据库写入失败不阻断选股主流程 |

候选上下文模块也支持 24 小时文件缓存，但 DSA 集成默认关闭其独立新闻/公告抓取，改用 DSA 自己的资讯、基本面和实时行情链路，避免同一候选重复请求两套数据源。

## 两类策略的边界

DSA 中存在两类用途不同的策略文件：

| 位置 | 解决的问题 | 加载方 | 执行阶段 |
| --- | --- | --- | --- |
| `src/services/screening/strategies/*.yaml` | 从全市场筛出哪些候选 | `src/services/screening/strategy.py` | 快照过滤、因子评分、风险和排序 |
| `strategies/*.yaml` | 对单只股票如何分析和形成结论 | `src/agent/skills/base.py` | DSA Agent/报告分析 |

当前筛选策略 `martin_momentum_pullback` 会把 `bull_trend` 与 `shrink_pullback` 作为后续单股分析 skill；这些 skill 不是可独立启动的旧筛选策略。Web 的“进一步深度分析”会显式携带该映射，两个目录仍使用不同 Schema 和 loader。

## DSA 原生能力复用

- 行情：大样本日 K 使用筛选模块本地缓存与轻量多源链，全部失败时才调用 DSA `DataFetcherManager`；最终候选继续补 DSA 实时行情，并按国内量化模式执行 AkShare/Baostock 复核。
- 基本面与资讯：最终候选复用 DSA 基本面上下文和 `SearchService`；资本流向来自 DSA 基本面上下文，重要公告/业绩/减持事件调用 DSA `search_stock_events`，热点消息搜索沿用其数据源优先级、时效过滤、缓存和同请求合并，仅将真实供应商调用隔离到可终止子进程，不重复维护独立资讯入口。
- 模型：沿用 DSA LiteLLM 模型、渠道、fallback、base URL、额外 headers、超时和 token 配置。
- 任务与页面：复用 DSA 后台任务队列、Web 轮询和桌面端同源 Web 资源。
- 存储与后续分析：运行结果写入 DSA 数据库；候选可进入 DSA 原生单股分析并携带策略 skill。

对照固定参考提交，快照、日 K、美股、行业/概念、热点、候选新闻/公告/资金流、字段标准化、过滤、评分、风险、排序和数据源熔断等原始数据与选股能力均已纳入；其中公告/事件和资金流在 DSA 编排层分别接入原生事件搜索与基本面上下文。参考项目另外提供独立 CLI/server、JSON 文件 store、报告渲染、doctor、运行/数据源历史和 T+N 评估：本实现只吸收 DSA 确实缺少的运行历史与数据源历史，并接到 DSA 数据库；CLI/server 不重复建设。历史 AI 报告的 T+N 评估与表现统计继续复用 DSA `BacktestService`；新增 Martin 组合模拟只读取本地日线研究产物，不写该服务的数据表，两种口径在命令、文档和存储上保持隔离。实时 source health 已在 `/status` 返回，历史稳定性由 `/source-history` 补齐。

## 收益

1. 选股服务、策略、API、Web 和打包脚本在同一版本中演进，避免契约漂移。
2. 服务层只有一套原生调用路径，状态探针和业务请求反映相同实现。
3. Docker 与桌面产物直接收集同一份模块和策略资源，部署结果更一致。
4. 数据源降级、评分和策略变化可以在主仓库完成端到端审查与回归。
5. 来源 commit、许可证和逐文件归因明确，便于后续选择性同步上游修复。

## 风险与控制

| 风险 | 影响 | 控制措施 |
| --- | --- | --- |
| 主仓库维护面扩大 | 数据源或策略问题由 DSA 直接承担 | 模块边界、契约测试和 CI 打包探针共同约束 |
| 与参考项目逐渐分叉 | 上游修复不能直接覆盖 | 固定参考 revision，逐模块比较并选择性移植 |
| 数据源限流或字段变化 | 快照、热点或日 K 降级 | timeout、retry、source health 与 last-good cache |
| LLM 超时或格式异常 | 重排不可用或解释字段缺失 | 非结构化响应继续尝试备用模型；全部失败时保留因子排序，并返回尝试模型与失败原因 |
| 结果轮换扩大候选差异 | 临界候选可能因浏览器不同而变化 | 仅轮换近分尾部位，保持硬过滤、风险否决、分值和头部候选不变；无种子时关闭 |
| 缓存目录变化 | 升级后旧缓存不会自动复用 | 新目录独立为 `data/screening`；升级前按需备份 |
| 运行历史增长 | 完整候选结果会增加数据库体积 | 历史接口默认只读摘要，运维可按现有数据库备份/保留策略管理 |
| 配置与 API 更名 | 旧自动化需同步调整 | 在发布说明明确 `SCREENING_ENABLED` 与 `/api/v1/screening` |
| 许可证归因遗漏 | 发布合规风险 | 保留 LICENSE、THIRD_PARTY_NOTICES 和衍生文件头 |

选股结果仅用于研究和辅助判断，不构成投资建议，也不保证收益或数据完整性。

## 更新参考实现

AlphaSift 是参考来源，不是自动同步源。更新时应：

1. 记录目标 commit 和许可证变化；
2. 比较 `src/services/screening/` 的 DSA 特有修改，按模块选择性移植；
3. 更新衍生文件头、`REFERENCE_REVISION` 和 `THIRD_PARTY_NOTICES.md`；
4. 检查 pipeline、API/Web 字段、数据源降级、策略资源与冻结打包；
5. 更新本文档和 `docs/CHANGELOG.md`，完成后端、Web、Docker/桌面验证。

## 回滚

- 业务回滚：设置 `SCREENING_ENABLED=false` 并重启；普通个股分析、报告、通知和问股不受影响。
- 代码回滚：revert 引入选股引擎的提交并重建后端、Docker 与桌面产物。
- 数据回滚：如需保留选股缓存和运行历史，先备份 `data/screening/` 与 DSA 数据库；代码回滚不会主动删除 `screening_runs` 用户数据。
