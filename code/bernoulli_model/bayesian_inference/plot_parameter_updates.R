
# ベルヌーイモデル -------------------------------------------------------------

# chapter 3.2.1
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布(ベルヌーイ分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 0.25


### 観測データの設定 -----

# (ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# 観測データを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)


#### 変数の設定 -----

# x軸の範囲を設定
x_min <- 0
x_max <- 1
cat('x size:', x_min, x_max)

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, by = 1)


# μ軸の範囲を設定
mu_min <- x_min
mu_max <- x_max
cat('μ size:', mu_min, mu_max)

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 1001)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
a <- 1
b <- 1

# 予測分布のパラメータを計算:式(3.19)
mu_star <- a / (a + b)

# 受け皿を初期化
trace_a_i  <- rep(NA, times = N+1)
trace_b_i  <- rep(NA, times = N+1)
trace_mu_i <- rep(NA, times = N+1)

# 初期値を記録
trace_a_i[1]  <- a
trace_b_i[1]  <- b
trace_mu_i[1] <- mu_star

# ベイズ推論の処理
for(n in 1:N){
  
  # 観測データを取得
  x <- x_n[n]
  
  # 事後分布のパラメータを更新:式(3.15)
  a <- a + x
  b <- b + 1 - x
  
  # 予測分布のパラメーターを更新:式(3.19)
  mu_star <- a / (a + b)
  
  # 更新値を記録
  trace_a_i[n+1]  <- a
  trace_b_i[n+1]  <- b
  trace_mu_i[n+1] <- mu_star
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
a <- 1
b <- 1

# 事後分布のパラメータを更新:式(3.15)
trace_a_i <- c(a, cumsum(x_n) + a)
trace_b_i <- c(a, 1:N - cumsum(x_n) + a)

# 予測分布のパラメーターを更新:式(3.19)
trace_mu_i <- trace_a_i / (trace_a_i + trace_b_i)


### 推移の作図 -----

# 観測データを格納
anim_sample_df <- tibble::tibble(
  n = 0:N,       # データ番号
  x = c(NA, x_n) # 観測値
)


#### 事後分布の作図 -----

# 事後分布の確率密度を計算
anim_posterior_df <- tidyr::expand_grid(
  n  = 0:N,   # 試行回数
  mu = mu_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    a    = trace_a_i[n+1], # パラメータ
    b    = trace_b_i[n+1], # パラメータ
    dens = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
  )

# 事後分布のラベルを作成
anim_param_df <- tibble::tibble(
  n = 0:N, 
  a = trace_a_i, 
  b = trace_b_i, 
  posterior_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "mu[truth] == ", round(mu_truth, digits = 2), ", ", 
    "hat(a) == ", round(a, digits = 1), ", ", 
    "hat(b) == ", round(b, digits = 1), 
    ")"
  )
)


# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_vline(
    mapping = aes(xintercept = mu_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = anim_posterior_df, 
    mapping = aes(x = mu, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x, y = -Inf, color = "sample"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = posterior_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -1
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = mu_truth, 
      labels    = expression(mu[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", "purple", "hotpink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        linewidth = c(0.5, 0.5, NA), 
        linetype  = c("dashed", "solid", NA), 
        size      = c(NA, NA, 5), 
        shape     = c(NA, NA, "circle"))
    )
  ) + 
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Beta distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "μ", 
    y = "density"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/bernoulli_model/parameter_updates/posterior.mp4")
)


#### 予測分布の作図 -----

# 生成分布の確率を計算
anim_model_df <- tidyr::expand_grid(
  n = 0:N, # データ番号
  tibble::tibble(
    x    = x_vec, # 確率変数
    mu   = mu_truth, # 成功確率パラメータ
    prob = dbinom(x = x, size = 1, prob = mu) # 確率
  )
) # 試行ごとに分布を複製

# 予測分布の確率を計算
anim_predict_df <- tidyr::expand_grid(
  n = 0:N,  # 試行回数
  x = x_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    mu   = trace_mu_i[n+1], # 成功確率パラメータ
    prob = dbinom(x = x, size = 1, prob = mu) # 確率
  )


# 予測分布のラベルを作成
anim_param_df <- tibble::tibble(
  n  = 0:N, 
  mu = trace_mu_i, 
  predict_param_lbl = paste0(
    "list(", 
    "N == ", n, ", ", 
    "mu[truth] == ", round(mu_truth, digits = 2), ", ", 
    "hat(mu)[star] == ", round(mu, digits = 5), 
    ")"
  )
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(
    data    = anim_model_df, 
    mapping = aes(x = x, y = prob, color = "model"), 
    stat = "identity", position = "identity",
    fill = NA, linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_bar(
    data    = anim_predict_df, 
    mapping = aes(x = x, y = prob, color = "predict"), 
    stat = "identity", position = "identity", 
    fill = "purple", alpha = 0.5, linetype = "blank"
  ) + # 予測分布
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x, y = -Inf, color = "sample"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = predict_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -0.5
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_color_manual(
    breaks = c("model", "predict", "sample"), 
    values = c("red", NA, "hotpink"), 
    labels = c("true model", "predict distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        size      = c(NA, NA, 5), 
        shape     = c(NA, NA, "circle"), 
        linewidth = c(0.5, 0, NA), 
        linetype  = c("dashed", "blank", NA)
      )
    )
  ) + 
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Bernoulli distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "x", 
    y = "probability"
  )

# 動画を作成
gganimate::animate(
  plot = predict_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(file = "figure/bernoulli_model/parameter_updates/predict.mp4")
)


#### 観測データと分布の関係 -----

# 一時書き出し先を指定
dir_path <- "figure/tmp_folder"


# 確率軸の範囲を設定
u <- 0.05
prob_max <- dbinom(
  x    = ifelse(
    test = trace_mu_i >= 0, 
    yes  = 1, 
    no   = 0
  ), # 最頻値
  size = 1, 
  prob = trace_mu_i
) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
prob_max

# 確率密度軸の範囲を設定
u <- 0.05
dens_max <- dbeta(
  x      = (trace_a_i - 1) / (trace_a_i + trace_b_i - 2), # 最頻値
  shape1 = trace_a_i, 
  shape2 = trace_b_i
) |> 
  max(na.rm = TRUE) |> # (a=b=1の場合は除く)
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
dens_max

# 試行ごとに作図
for(i in 1:(N+1)) {
  
  # 値を取得
  n       <- i - 1  # データ番号
  x       <- x_n[n] # 観測値
  a       <- trace_a_i[i]  # パラメータ
  b       <- trace_b_i[i]  # パラメータ
  mu_star <- trace_mu_i[i] # 成功確率パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x = x # 観測値
  )
  
  ##### 観測データの作図 -----
  
  # 生成分布の期待値を計算
  E_x <- mu_truth
  
  # 観測データの標本平均を計算
  bar_x = mean(x_n[0:n])
  
  # 生成分布の確率を計算
  model_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    prob = c(1-mu_truth, mu_truth) # 確率
  )
  
  # 観測データを集計
  obs_df <- tibble::tibble(
    x        = x_vec, # 観測値
    freq     = c(n-sum(x_n[0:n]), sum(x_n[0:n])), # 度数
    rel_freq = freq / n # 相対度数
  )
  
  # 生成分布のラベルを作成
  model_param_lbl <- paste0(
    "list(", 
    "N == ", n, ", ", 
    "mu[truth] == ", round(mu_truth, digits = 2), ", ", 
    "paste(E(x) == mu[truth], {} == ", round(E_x, digits = 2), ")", ", ", 
    "bar(x) == ", round(bar_x, digits = 5), 
    ")"
  ) |> 
    parse(text = _)
  
  # (目盛の共通化用)
  dummy_graph <- ggplot() + 
    geom_bar(
      data    = model_df, 
      mapping = aes(x = x, y = prob), 
      stat = "identity", position = "identity"
    ) + # 真の分布
    coord_cartesian(
      ylim = c(0, prob_max) # (目盛の共通化用)
    )
  built     <- ggplot_build(dummy_graph) # 図情報を取得
  prob_vals <- built$layout$panel_params[[1]]$y$breaks         # 確率軸目盛を取得
  freq_vals <- prob_vals * ifelse(test = n>0, yes = n, no = 1) # 度数軸目盛に変換
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = bar_x), 
      color = "hotpink", linewidth = 1, linetype = "dashed"
    ) + # 標本平均
    geom_bar(
      data    = model_df, 
      mapping = aes(x = x, y = prob, color = "model"), 
      stat = "identity", position = "identity",
      fill = NA, linewidth = 1, linetype = "dashed"
    ) + # 真の分布
    geom_bar(
      data    = obs_df, 
      mapping = aes(x = x, y = rel_freq, color = "sample"), 
      stat = "identity", position = "identity", 
      fill = "hotpink", alpha = 0.5, linetype = "blank"
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      breaks = x_vec, minor_breaks = FALSE, 
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(E_x, bar_x), 
        labels    = c(expression(E(x)), expression(bar(x)))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      breaks = prob_vals, # (目盛の共通化用)
      sec.axis = sec_axis(
        transform = ~ .*ifelse(test = n>0, yes = n, no = 1), 
        breaks    = freq_vals, # (目盛の共通化用)
        labels    = scales::label_number(accuracy = 0.01), # (描画領域のズレの対策用)
        name      = "frequency"
      )
    ) + # 度数軸目盛
    scale_color_manual(
      breaks = c("model", "sample"), 
      values = c("red", NA), 
      labels = c("true model", "observation data"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(
        override.aes = list(
          linewidth = c(0.5, 0), 
          linetype  = c("dashed", "blank")
        )
      )
    ) + 
    coord_cartesian(
      xlim = c(x_min-0.5, x_max+0.5), # (目盛の共通化用)
      ylim = c(0, prob_max) # (目盛の共通化用)
    ) + 
    labs(
      title = "Bernoulli distribution", 
      subtitle = model_param_lbl, 
      x = expression(x), 
      y = "probability"
    )
  
  ##### 事後分布の作図 -----
  
  # 事後分布の期待値を計算
  E_mu <- a / (a + b)
  
  # 事後分布の確率を計算
  posterior_df <- tibble::tibble(
    mu   = mu_vec, # 確率変数
    dens = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
  )
  
  # 事後分布のラベルを作成
  posterior_param_lbl <- paste0(
    "list(", 
    "hat(a) == ", round(a, digits = 1), ", ", 
    "hat(b) == ", round(b, digits = 1), ", ", 
    "paste(E(mu) == frac(a, a+b), {} == ", round(E_mu, digits = 5), ")", 
    ")"
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_mu), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_line(
      data    = posterior_df, 
      mapping = aes(x = mu, y = dens, color = "posterior"), 
      linewidth = 1
    ) + # 事後分布
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth, E_mu), 
        labels    = c(expression(mu[truth]), expression(E(mu)))
      ) # パラメータラベル
    ) + 
    scale_color_manual(
      breaks = c("model", "posterior"), 
      values = c("red", "purple"), 
      labels = c("true parameter", "posterior distribution"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(x_min-0.5, x_max+0.5), # (目盛の共通化用)
      ylim = c(0, dens_max)
    ) + 
    labs(
      title = "Beta distribution", 
      subtitle = posterior_param_lbl, 
      x = expression(mu), 
      y = "density"
    )
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x <- mu_star
  
  # 予測分布の確率を計算
  predict_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    prob = c(1-mu_star, mu_star) # 確率
  )
  
  # 予測分布のラベルを作成
  predict_param_lbl <- paste0(
    "list(", 
    "hat(mu)[star] == ", round(mu_star, digits = 5), ", ", 
    "paste(E(x) == hat(mu)[star], {} == ", round(E_x, digits = 5), ")", 
    ")"
  ) |> 
    parse(text = _)
  
  # 予測分布のアニメーションを作図
  predict_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_x), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_bar(
      data    = model_df, 
      mapping = aes(x = x, y = prob, color = "model"), 
      stat = "identity", position = "identity",
      fill = NA, linewidth = 1, linetype = "dashed"
    ) + # 真の分布
    geom_bar(
      data    = predict_df, 
      mapping = aes(x = x, y = prob, color = "predict"), 
      stat = "identity", position = "identity", 
      fill = "purple", alpha = 0.5, linetype = "blank"
    ) + # 予測分布
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      breaks = x_vec, minor_breaks = FALSE, 
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth, E_x), 
        labels    = c(expression(mu[truth]), expression(E(x)))
      ) # パラメータラベル
    ) + 
    scale_color_manual(
      breaks = c("model", "predict"), 
      values = c("red", NA), 
      labels = c("true model", "predict distribution"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(
        override.aes = list(
          linewidth = c(0.5, 0), 
          linetype  = c("dashed", "blank")
        )
      )
    ) + 
    coord_cartesian(
      xlim = c(x_min-0.5, x_max+0.5), # (目盛の共通化用)
      ylim = c(0, prob_max)
    ) + 
    labs(
      title = "Negative Binomial distribution", 
      subtitle = predict_param_lbl, 
      x = expression(x), 
      y = "probability"
    )
  
  ##### 対応関係の作図 -----
  
  # 図タイトルを指定
  title_graph <- cowplot::ggdraw() + 
    cowplot::draw_label(
      label = "Bayesian inference", 
      size = 20
    )
  
  # グラフを並べて描画
  comb_graph <- cowplot::plot_grid(
    model_graph, posterior_graph, predict_graph, 
    nrow = 3, ncol = 1, 
    align = "hv" # (目盛の共通化用)
  )
  final_graph <- cowplot::plot_grid(
    title_graph, comb_graph, 
    nrow = 2, ncol = 1, 
    rel_heights = c(0.1, 1)
  )
  
  ##### グラフの出力 -----
  
  # 画像ファイルを書出
  file_path <- paste0(dir_path, "/", stringr::str_pad(n, width = nchar(N), pad = "0"), ".png")
  ggplot2::ggsave(
    filename = file_path, plot = comb_graph, 
    width = 9, height = 12, units = "in", dpi = 100
  )
  
  # 途中経過を表示
  message("\r", n, " / ", N, appendLF = FALSE)
}

# 動画を作成
paste0(dir_path, "/", stringr::str_pad(0:N, width = nchar(N), pad = "0"), ".png") |> # ファイルパスを作成
  magick::image_read() |> # pngファイルを読込
  magick::image_animate(fps = 1, dispose = "previous") |> # gifファイルを作成
  magick::image_write_video(path = "figure/bernoulli_model/parameter_updates/observation.mp4", framerate = 10) -> tmp_path # mp4ファイルを書出


