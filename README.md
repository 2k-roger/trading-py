# trading-py — 量化交易回测 & 实盘系统

基于 Python + ccxt 的 Binance 合约量化交易系统，支持**策略回测**、**参数优化**和**实时模拟交易**（WebSocket）。

---

## 功能特性

- 🧪 **高性能回测引擎** — 向量化指标计算 + numpy 加速逐 bar 模拟（~8000 bars/s）
- 📈 **多种内置策略** — EMA 金叉/死叉、均值回归、突破策略，双向做多做空
- ⚡ **杠杆合约模拟** — U 本位永续合约，逐仓模式，爆仓/保证金/手续费全量模拟
- 🎯 **完整风控体系** — 固定风险仓位计算、ATR 移动止损、初始止损/移动止损区分
- 📊 **19 项绩效指标** — 夏普比率、索提诺、卡尔玛、最大回撤、盈亏比、胜率等
- 📉 **四面板可视化** — 权益曲线、回撤图、价格+交易标记、每笔 PnL 分布
- 🔌 **策略可插拔** — `@register` 装饰器注册，新策略只需一个文件
- 🔴 **WebSocket 实盘/模拟交易** — ccxt.pro 订阅 Binance 实时 K 线推送
- 🗂️ **数据缓存** — OHLCV 数据本地 CSV 缓存，免重复下载
- 🌐 **中文输出** — 所有日志、指标、出场原因均已汉化
- ⚙️ **YAML 配置** — 全局默认配置 + 运行时覆盖，灵活切换策略和参数

---

## 环境要求

- Python ≥ 3.10
- pip

依赖库：

| 库 | 版本 | 用途 |
|---|---|---|
| ccxt | ≥ 4.0 | 交易所 REST API + WebSocket |
| pandas | ≥ 2.0 | 数据处理与指标计算 |
| numpy | ≥ 1.24 | 数值计算 |
| matplotlib | ≥ 3.7 | 回测可视化 |
| pyyaml | ≥ 6.0 | 配置文件解析 |

---

## 快速开始

### 1. 克隆 & 安装

```bash
git clone git@github.com:2k-roger/trading-py.git
cd trading-py

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行回测

```bash
python examples/run_backtest.py
```

输出示例：

```
已加载 ETH/USDT 1h K线数据：2,134 根
时间范围：2026-04-01 00:00:00 → 2026-06-29 08:00:00
交易策略：ema_crossover_v2
交易杠杆：15x

============================================================
  指标                                  数值
------------------------------------------
  总收益率                            +38.92%
  年化收益率 (CAGR)                   +156.2%
  夏普比率 (Sharpe)                      4.33
  最大回撤                              -8.12%
  胜率                                  52.3%
  盈亏比 (Profit Factor)                 2.15
  ...
============================================================

初始资金：$1000.00
最终权益：$1389.20
净盈亏：  +$389.20 (+38.9%)

序号  入场时间              出场时间              方向    盈亏%    出场原因
----------------------------------------------------------------------
1     2026-04-01 06:00:00   2026-04-01 12:00:00   做空    +1.23%   移动止损
2     2026-04-02 08:00:00   2026-04-02 20:00:00   做多    +2.45%   移动止损
...
```

### 3. 实时模拟交易（WebSocket）

```bash
python examples/live_trading.py
```

连接 Binance WebSocket，实时推送 K 线，策略自动执行。日志同时输出到控制台和 `logs/live_trading.log`。

```bash
# 调试模式 — 查看每根 K 线的权益快照和止损变动
python examples/live_trading.py --log-level DEBUG
```

### 4. 参数优化

```bash
python examples/optimize.py
```

对 EMA V2、均值回归、突破策略进行网格搜索，输出最优配置。结果保存到 `data/opt_results.json`。

---

## 项目结构

```
trading-py/
├── config/
│   └── default.yaml               # 全局默认配置
├── src/
│   ├── models.py                  # 数据模型（Trade, Position, Metrics 等）
│   ├── config.py                  # YAML 配置加载与合并
│   ├── data/
│   │   └── loader.py              # ccxt 获取 + CSV 缓存
│   ├── strategies/
│   │   ├── base.py                # Strategy 抽象基类
│   │   ├── registry.py            # @register 装饰器注册系统
│   │   ├── ema_crossover_atr.py   # EMA V1：金叉做多 + ATR 止损
│   │   └── ema_crossover_v2.py    # EMA V2：双向 + 趋势过滤 + 紧止损
│   ├── backtest/
│   │   └── engine.py              # 回测引擎（含 FastBar 加速）
│   ├── risk/
│   │   └── manager.py             # 仓位计算、爆仓价、手续费
│   ├── metrics/
│   │   └── calculator.py          # 19 项绩效指标
│   ├── visualization/
│   │   └── plotter.py             # 四面板可视化图表
│   └── live/
│       └── __init__.py            # 实盘交易占位
├── examples/
│   ├── run_backtest.py            # 一键回测
│   ├── optimize.py                # 参数网格搜索优化
│   ├── live_trading.py            # WebSocket 实时模拟交易
│   └── analyze_data.py            # 市场数据分析
├── data/                          # OHLCV 缓存（.gitignore）
├── logs/                          # 交易日志（.gitignore）
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 配置说明

编辑 `config/default.yaml`：

```yaml
# 数据
exchange: binance
symbol: ETH/USDT          # 交易对
timeframe: 1h             # K 线周期 (1m/5m/15m/1h/4h/1d)
start_date: 2026-04-01   # 回测起始日期
end_date: 2026-06-29      # 回测结束日期

# 资金与杠杆
initial_capital: 1000.0   # 初始本金 (USDT)
leverage: 15              # 杠杆倍数
margin_mode: isolated     # 逐仓模式
maintenance_margin_pct: 0.005  # 维持保证金率 (ETH: 0.5%)

# 费用
commission_pct: 0.0004    # 0.04% taker 手续费
slippage_pct: 0.0001      # 0.01% 滑点

# 风控
position_sizing: fixed_risk   # 固定风险仓位
risk_per_trade_pct: 0.01      # 每笔风险 1% 本金

# 策略
strategy_name: ema_crossover_v2
strategy_params:
  ema_short: 21
  ema_long: 55
  ema_trend: 100
  atr_period: 14
  atr_mult: 0.25
  tp_mult: 0.0
```

也可以在代码中覆盖配置：

```python
config = ConfigLoader.load(overrides={
    'strategy_name': 'ema_crossover_atr',
    'leverage': 10,
})
```

---

## 内置策略

### EMA Crossover V1 (`ema_crossover_atr`)

经典 EMA 金叉趋势跟踪策略。

| 参数 | 默认 | 说明 |
|---|---|---|
| `ema_short` | 9 | 快线周期 |
| `ema_long` | 21 | 慢线周期 |
| `atr_period` | 14 | ATR 计算周期 |
| `atr_multiplier` | 3.0 | ATR 止损倍数 |

入场：EMA 快线上穿慢线 → 做多。止损：ATR × 倍数。仅做多。

### EMA Crossover V2 (`ema_crossover_v2`) ⭐ 推荐

双向趋势跟踪 + 趋势过滤 + 紧止损，针对当前行情优化。

| 参数 | 默认 | 说明 |
|---|---|---|
| `ema_short` | 21 | 快线周期 |
| `ema_long` | 55 | 慢线周期 |
| `ema_trend` | 100 | 趋势过滤 MA |
| `atr_period` | 14 | ATR 周期 |
| `atr_mult` | 0.25 | ATR 止损倍数（紧止损） |
| `tp_mult` | 0.0 | 止盈倍数（0 = 不设止盈） |

**最佳参数**（2026-04 ~ 2026-06 ETH/USDT 1h）：
- 15x 杠杆，+38.9%，夏普 4.33，0 次爆仓
- 做多 + 做空双向交易，趋势 MA 过滤噪音
- ATR×0.25 极紧止损实现快速锁定利润

---

## 编写自定义策略

只需继承 `Strategy` 并注册：

```python
from src.strategies.base import Strategy
from src.strategies.registry import register
from src.models import TradeSetup, Position

@register('my_strategy')
class MyStrategy(Strategy):
    def compute_indicators(self, df):
        df = df.copy()
        df['ma'] = df['close'].rolling(20).mean()
        return df

    def on_bar(self, df, idx, position):
        if position is not None:
            return TradeSetup(action='none')  # 引擎处理出场

        # 入场逻辑
        row = df.iloc[idx]
        if row['close'] > row['ma']:
            return TradeSetup(
                action='enter_long',
                stop_loss=row['close'] * 0.98,
                take_profit=row['close'] * 1.04,
            )
        return TradeSetup(action='none')

    def get_trailing_stop(self, position, bar):
        # 可选：实现动态止损
        return position.stop_loss
```

然后在回测脚本中 `import` 你的策略文件，`@register` 会自动生效。配置中指定 `strategy_name: my_strategy` 即可使用。

---

## 回测引擎核心逻辑

每根 K 线，引擎按优先级检查：

1. **爆仓** → 价格触及强平线？亏损全部保证金
2. **止损** → 区分初始止损 (`stop_loss`) 和移动止损 (`trailing_stop`)
3. **止盈** → 价格触及目标价？
4. **入场** → 无持仓时调用 `strategy.on_bar()`
5. **移动止损更新** → ratchet 机制：做多只上移，做空只下移
6. **权益快照** → 现金 + 未实现盈亏

资金模型：

```
开仓：现金 -= 保证金 + 开仓手续费
平仓：现金 += max(0, 保证金 + 盈亏 - 平仓手续费)
爆仓：亏损 = 全部保证金（逐仓隔离）
```

---

## 绩效指标

系统计算 19 项指标（中文输出）：

| 指标 | 说明 |
|---|---|
| 总收益率 | (期末权益/初始本金 - 1) × 100% |
| 年化收益率 (CAGR) | 几何年化复利 |
| 夏普比率 | (年化收益 - 无风险利率) / 年化波动率 |
| 索提诺比率 | 仅用下行波动率 |
| 卡尔玛比率 | CAGR / 最大回撤 |
| 最大回撤 | 权益从峰值回落的最大幅度 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 总盈利 / 总亏损 |
| 期望值 | 每笔交易的平均期望收益 (%) |
| 爆仓次数 | 强平次数 |

---

## 实时交易

`examples/live_trading.py` 使用 ccxt.pro WebSocket：

- 拉取 300 根历史 K 线预热指标
- 订阅 `ETHUSDT@kline_1h`
- Binance 主动推送每根 K 线，闭合时触发策略
- 交易日志写入 `logs/live_trading.log`
- `Ctrl+C` 安全退出

与回测完全相同的策略接口：`Strategy.on_bar()` 和 `RiskManager` 可直接复用到实盘。

---

## 常用命令

```bash
# 回测
python examples/run_backtest.py

# 实时交易
python examples/live_trading.py
python examples/live_trading.py --log-level DEBUG

# 参数优化（可能需要几分钟）
python examples/optimize.py

# 数据分析
python examples/analyze_data.py
```

---

## 设计决策

| 决策 | 理由 |
|---|---|
| 向量化指标 + 逐 bar 模拟 | 动态止损是有状态的，纯向量化不可行 |
| numpy FastBar 加速 | 去除 pandas 开销，5.5x 提速 |
| 装饰器策略注册 | 显式注册，零魔法，易调试 |
| 单 models.py | 所有 dataclass 集中管理，避免循环引用 |
| 逐仓模式默认 | 爆仓仅损失单仓保证金，风险隔离 |
| 固定百分比滑点 | 合理的惩罚模型，无需订单簿级模拟 |

---

## 许可

MIT
