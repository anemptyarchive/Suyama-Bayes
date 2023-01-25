
# 3.2.2 カテゴリ分布の学習と予測 -------------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(MCMCpack)
library(gganimate)

# チェック用
library(ggplot2)


# 三角図の準備 ------------------------------------------------------------------

### ・三角座標用 -----

# 軸目盛の位置を指定
axis_vals <- seq(from = 0, to = 1, by = 0.1)

# 枠線用の値を作成
ternary_axis_df <- tibble::tibble(
  y_1_start = c(0.5, 0, 1),         # 始点のx軸の値
  y_2_start = c(0.5*sqrt(3), 0, 0), # 始点のy軸の値
  y_1_end = c(0, 1, 0.5),           # 終点のx軸の値
  y_2_end = c(0, 0, 0.5*sqrt(3)),   # 終点のy軸の値
  axis = c("x_1", "x_2", "x_3")     # 元の軸
)

# グリッド線用の値を作成
ternary_grid_df <- tibble::tibble(
  y_1_start = c(
    0.5 * axis_vals, 
    axis_vals, 
    0.5 * axis_vals + 0.5
  ), # 始点のx軸の値
  y_2_start = c(
    sqrt(3) * 0.5 * axis_vals, 
    rep(0, times = length(axis_vals)), 
    sqrt(3) * 0.5 * (1 - axis_vals)
  ), # 始点のy軸の値
  y_1_end = c(
    axis_vals, 
    0.5 * axis_vals + 0.5, 
    0.5 * rev(axis_vals)
  ), # 終点のx軸の値
  y_2_end = c(
    rep(0, times = length(axis_vals)), 
    sqrt(3) * 0.5 * (1 - axis_vals), 
    sqrt(3) * 0.5 * rev(axis_vals)
  ), # 終点のy軸の値
  axis = c("x_1", "x_2", "x_3") |> 
    rep(each = length(axis_vals)) # 元の軸
)

# 軸ラベル用の値を作成
ternary_axislabel_df <- tibble::tibble(
  y_1 = c(0.25, 0.5, 0.75),               # x軸の値
  y_2 = c(0.25*sqrt(3), 0, 0.25*sqrt(3)), # y軸の値
  label = c("phi[1]", "phi[2]", "phi[3]"),      # 軸ラベル
  h = c(3, 0.5, -2),  # 水平方向の調整用の値
  v = c(0.5, 3, 0.5), # 垂直方向の調整用の値
  axis = c("x_1", "x_2", "x_3") # 元の軸
)

# 軸目盛ラベル用の値を作成
ternary_ticklabel_df <- tibble::tibble(
  y_1 = c(
    0.5 * axis_vals, 
    axis_vals, 
    0.5 * axis_vals + 0.5
  ), # x軸の値
  y_2 = c(
    sqrt(3) * 0.5 * axis_vals, 
    rep(0, times = length(axis_vals)), 
    sqrt(3) * 0.5 * (1 - axis_vals)
  ), # y軸の値
  label = c(
    rev(axis_vals), 
    axis_vals, 
    rev(axis_vals)
  ), # 軸目盛ラベル
  h = c(
    rep(1.5, times = length(axis_vals)), 
    rep(1.5, times = length(axis_vals)), 
    rep(-0.5, times = length(axis_vals))
  ), # 水平方向の調整用の値
  v = c(
    rep(0.5, times = length(axis_vals)), 
    rep(0.5, times = length(axis_vals)), 
    rep(0.5, times = length(axis_vals))
  ), # 垂直方向の調整用の値
  angle = c(
    rep(-60, times = length(axis_vals)), 
    rep(60, times = length(axis_vals)), 
    rep(0, times = length(axis_vals))
  ), # ラベルの表示角度
  axis = c("x_1", "x_2", "x_3") |> 
    rep(each = length(axis_vals)) # 元の軸
)


### ・等高線用 -----

# 三角座標の値を作成
y_1_vals <- seq(from = 0, to = 1, length.out = 151)
y_2_vals <- seq(from = 0, to = 0.5*sqrt(3), length.out = 150)

# 格子点を作成
y_mat <- tidyr::expand_grid(
  y_1 = y_1_vals, 
  y_2 = y_2_vals
) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# 3次元変数に変換
phi_mat <- tibble::tibble(
  phi_2 = y_mat[, 1] - y_mat[, 2] / sqrt(3), 
  phi_3 = 2 * y_mat[, 2] / sqrt(3)
) |> # 元の座標に変換
  dplyr::mutate(
    phi_2 = dplyr::if_else(phi_2 >= 0 & phi_2 <= 1, true = phi_2, false = as.numeric(NA)), 
    phi_3 = dplyr::if_else(phi_3 >= 0 & phi_3 <= 1 & !is.na(phi_2), true = phi_3, false = as.numeric(NA)), 
    phi_1 = 1 - phi_2 - phi_3, 
    phi_1 = dplyr::if_else(phi_1 >= 0 & phi_1 <= 1, true = phi_1, false = as.numeric(NA))
  ) |> # 範囲外の値をNAに置換
  dplyr::select(phi_1, phi_2, phi_3) |> # 順番を変更
  as.matrix() # マトリクスに変換


# ベイズ推論の実装 -------------------------------------------------------------------

### ・生成分布(カテゴリ分布)の設定 -----

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)

# 次元数を設定:(固定)
K <- length(pi_truth_k)


# 真の分布を格納
model_df <- tibble::tibble(
  k = 1:K, # 次元番号
  prob = pi_truth_k # 確率
)


# パラメータラベル用の文字列を作成
model_param_text <- paste0("pi==(list(", paste0(pi_truth_k, collapse = ", "), "))")

# 真の分布を作図
ggplot() + 
  geom_bar(data = model_df, mapping = aes(x = k, y = prob, fill = "model"), 
           stat = "identity") + # 真の分布
  scale_fill_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # バーの色:(凡例表示用)
  scale_x_continuous(breaks = 1:K, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = parse(text = model_param_text), 
       x = "k", y = "probability")


### ・データの生成 -----

# データ数を指定
N <- 50


# カテゴリ分布に従うデータを生成
s_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) |> 
  t()

# 観測データを集計
freq_df <- tibble::tibble(
  k = 1:K, # 次元番号
  freq = colSums(s_nk) # 度数
)


# パラメータラベル用の文字列を作成
sample_param_text <- paste0(
  "list(", 
  "N==", N, "~(list(", paste0(freq_df[["freq"]], collapse = ", "), "))", 
  ", pi==(list(", paste0(pi_truth_k, collapse = ", "), "))", 
  ")"
)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_bar(data = freq_df, mapping = aes(x = k, y = freq/N, fill = "data"), 
           stat = "identity") + # 観測データ:(相対度数)
  geom_bar(data = model_df, mapping = aes(x = k, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = 1:K, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = parse(text = sample_param_text), 
       x = "k", y = "relative frequency")


### ・事前分布(ディリクレ分布)の設定 -----

# 事前分布のパラメータを指定
alpha_k <- rep(1, times = K)


# 事前分布を計算:式(2.48)
prior_df <- tibble::tibble(
  y_1 = y_mat[, 1], # x軸の値
  y_2 = y_mat[, 2], # y軸の値
  density = MCMCpack::ddirichlet(x = phi_mat, alpha = alpha_k) # 確率密度
) |> 
  dplyr::mutate(
    fill_flg = !is.na(rowSums(phi_mat)), 
    density = dplyr::if_else(fill_flg, true = density, false = as.numeric(NA))
  ) # 範囲外の値をNAに置換


# 真のパラメータを格納
parameter_df <- tibble::tibble(
  x = pi_truth_k[2] + 0.5 * pi_truth_k[3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_truth_k[3], # 三角座標への変換
)

# パラメータラベル用の文字列を作成
prior_param_text <- paste0("alpha==(list(", paste0(alpha_k, collapse = ", "), "))")

# 事前分布を作図
ggplot() + 
  geom_segment(data = ternary_grid_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50", linetype = "dashed") + # 三角図のグリッド線
  geom_segment(data = ternary_axis_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50") + # 三角図の枠線
  geom_text(data = ternary_ticklabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v, angle = angle)) + # 三角図の軸目盛ラベル
  geom_text(data = ternary_axislabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v), 
            parse = TRUE, size = 6) + # 三角図の軸ラベル
  geom_contour_filled(data = prior_df, 
                      mapping = aes(x = y_1, y = y_2, z = density, fill = ..level..), 
                      alpha = 0.8) + # 事前分布
  geom_point(data = parameter_df, mapping = aes(x = x, y = y, color = "param"), 
             shape = 4, size = 6) + # 真のパラメータ
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = NULL) + # x軸
  scale_y_continuous(breaks = c(0, 0.25*sqrt(3), 0.5*sqrt(3)), labels = NULL) + # y軸
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # 線の色:(凡例表示用)
  coord_fixed(ratio = 1, clip = "off") + # アスペクト比
  theme(axis.ticks = element_blank(), 
        panel.grid.minor = element_blank()) + # 図の体裁
  labs(title = "Dirichlet Distribution", 
       subtitle = parse(text = prior_param_text), 
       fill = "density", 
       x = "", y = "")


### ・事後分布(ディリクレ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.28)
alpha_hat_k <- colSums(s_nk) + alpha_k


# 事後分布を計算:式(2.48)
posterior_df <- tibble::tibble(
  y_1 = y_mat[, 1], # x軸の値
  y_2 = y_mat[, 2], # y軸の値
  density = MCMCpack::ddirichlet(x = phi_mat, alpha = alpha_hat_k) # 確率密度
) |> 
  dplyr::mutate(
    fill_flg = !is.na(rowSums(phi_mat)), 
    density = dplyr::if_else(fill_flg, true = density, false = as.numeric(NA))
  ) # 範囲外の値をNAに置換


# パラメータラベル用の文字列を作成
posterior_param_text <- paste0(
  "list(", 
  "N==", N, "~(list(", paste0(freq_df[["freq"]], collapse = ", "), "))", 
  ", hat(alpha)==(list(", paste0(alpha_hat_k, collapse = ", "), "))", 
  ")"
)

# 事後分布を作図
ggplot() + 
  geom_segment(data = ternary_grid_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50", linetype = "dashed") + # 三角図のグリッド線
  geom_segment(data = ternary_axis_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50") + # 三角図の枠線
  geom_text(data = ternary_ticklabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v, angle = angle)) + # 三角図の軸目盛ラベル
  geom_text(data = ternary_axislabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v), 
            parse = TRUE, size = 6) + # 三角図の軸ラベル
  geom_contour_filled(data = posterior_df, 
                      mapping = aes(x = y_1, y = y_2, z = density, fill = ..level..), 
                      alpha = 0.8) + # 事後分布
  geom_point(data = parameter_df, mapping = aes(x = x, y = y, color = "param"), 
             shape = 4, size = 6) + # 真のパラメータ
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = NULL) + # x軸
  scale_y_continuous(breaks = c(0, 0.25*sqrt(3), 0.5*sqrt(3)), labels = NULL) + # y軸
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # 線の色:(凡例表示用)
  coord_fixed(ratio = 1, clip = "off") + # アスペクト比
  theme(axis.ticks = element_blank(), 
        panel.grid.minor = element_blank()) + # 図の体裁
  labs(title = "Dirichlet Distribution", 
       subtitle = parse(text = posterior_param_text), 
       fill = "density", 
       x = "", y = "")


### ・予測分布(カテゴリ分布)の計算 -----

# 予測分布のパラメータを計算:式(3.31')
pi_star_hat_k <- alpha_hat_k / sum(alpha_hat_k)
pi_star_hat_k <- (colSums(s_nk) + alpha_k) / sum(colSums(s_nk) + alpha_k)


# 予測分布を格納
predict_df <- tibble::tibble(
  k = 1:K, # 次元番号
  prob = pi_star_hat_k # 確率
)


# パラメータラベル用の文字列を作成
predict_param_text <- paste0(
  "list(", 
  "N==", N, "~(list(", paste0(freq_df[["freq"]], collapse = ", "), "))", 
  ", pi[s]==(list(", paste0(round(pi_star_hat_k, digits = 2), collapse = ", "), "))", 
  ")"
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, mapping = aes(x = k, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = model_df, aes(x = k, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, predict ="purple"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict ="purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = 1:K, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = parse(text = predict_param_text), 
       x = "k", y = "probability")


# アニメーションによる学習推移の可視化 -----------------------------------------------------------------

### ・モデルの設定 -----

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)

# 事前分布のパラメータを指定
alpha_k <- rep(1, times = K)

# 次元数を設定:(固定)
K <- length(pi_truth_k)

# データ数(試行回数)を指定
N <- 100


### ・推論処理：for関数による処理 -----

# 事前分布(ディリクレ分布)を計算:式(2.48)
anime_posterior_df <- tibble::tibble(
  y_1 = y_mat[, 1], # x軸の値
  y_2 = y_mat[, 2], # y軸の値
  density = MCMCpack::ddirichlet(x = phi_mat, alpha = alpha_k) # 確率密度
) |> 
  dplyr::mutate(
    fill_flg = !is.na(rowSums(phi_mat)), 
    density = dplyr::if_else(fill_flg, true = density, false = as.numeric(NA)), # 範囲外の値をNAに置換
    param = paste0(
      "n=", 0, " (", paste0(rep(0, times = K), collapse = ", "), ")", 
      ", α=(", paste0(alpha_k, collapse = ", "), ")"
    ) |> 
      as.factor() # フレーム切替用のラベル
  )

# 初期値による予測分布のパラメーターを計算:式(3.31)
pi_star_k <- alpha_k / sum(alpha_k)

# 初期値による予測分布を格納
anime_predict_df <- tibble::tibble(
  k = 1:K, # 次元番号
  prob = pi_star_k, # 確率
  param = paste0(
      "N=", 0, " (", paste0(rep(0, times = K), collapse = ", "), ")", 
      ", π=(", paste0(round(pi_star_k, digits = 2), collapse = ", "), ")"
    ) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
s_nk <- matrix(NA, nrow = N, ncol = K)

# ベイズ推論
for(n in 1:N){
  
  # カテゴリ分布に従うデータを生成
  s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = pi_truth_k) |> 
    as.vector()
  
  # 事後分布のパラメータを更新:式(3.28)
  alpha_k <- s_nk[n, ] + alpha_k
  
  # 事後分布を計算:式(2.48)
  tmp_posterior_df <- tibble::tibble(
    y_1 = y_mat[, 1], # x軸の値
    y_2 = y_mat[, 2], # y軸の値
    density = MCMCpack::ddirichlet(x = phi_mat, alpha = alpha_k) # 確率密度
  ) |> 
    dplyr::mutate(
      fill_flg = !is.na(rowSums(phi_mat)), 
      density = dplyr::if_else(fill_flg, true = density, false = as.numeric(NA)), # 範囲外の値をNAに置換
      param = paste0(
        "n=", n, " (", paste0(colSums(s_nk), collapse = ", "), ")", 
        ", α=(", paste0(alpha_k, collapse = ", "), ")"
      ) |> 
        as.factor() # フレーム切替用のラベル
    )
  
  # 予測分布のパラメーターを更新:式(3.31)
  pi_star_k <- alpha_k / sum(alpha_k)
  
  # 予測分布を格納
  tmp_predict_df <- tibble::tibble(
    k = 1:K, # 次元番号
    prob = pi_star_k, # 確率
    param = paste0(
      "N=", n, " (", paste0(colSums(s_nk), collapse = ", "), ")", 
      ", π=(", paste0(round(pi_star_k, digits = 2), collapse = ", "), ")"
    ) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # n回目の結果を結合
  anime_posterior_df <- dplyr::bind_rows(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- dplyr::bind_rows(anime_predict_df, tmp_predict_df)
  
  # 途中経過を表示
  message("\r", n, " / ", N, appendLF = FALSE)
}

# 観測データを確認
colSums(s_nk)


### ・推論処理：tidyverseパッケージによる処理 -----

# カテゴリ分布に従うデータを生成
s_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) |> 
  t()


# レベルを設定用の文字列を作成
freq_vec <- rbind(rep(0, times = K), s_nk) |> 
  apply(MARGIN = 2, FUN = cumsum) |> # 累積和を計算
  apply(MARGIN = 1, FUN = paste0, collapse = ", ") # 試行ごとに結合
param_vec <- rbind(rep(0, times = K), s_nk) |> 
  apply(MARGIN = 2, FUN = cumsum) |> # 累積和を計算
  (\(.) {t(.) + alpha_k})() |> # 事後分布のパラメータを計算
  apply(MARGIN = 2, FUN = paste0, collapse = ", ") # 試行ごとに結合
posterior_level_vec <- paste0("n=", 0:N, " (", freq_vec, "), α=(", param_vec, ")")

# 試行ごとに事後分布を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  tibble::tibble(
    y_1 = y_mat[, 1], # x軸の値
    y_2 = y_mat[, 2]  # y軸の値
  )
) |> # 試行ごとに格子点を複製
  dplyr::group_by(n) |> # 試行ごとの計算用
  dplyr::mutate(
    # 事後分布のパラメータを計算:式(3.28)
    sum_s_k_lt = colSums(s_nk[0:unique(n), , drop = FALSE]) |> 
      list(), 
    alpha_k_lt = (colSums(s_nk[0:unique(n), , drop = FALSE]) + alpha_k) |> 
      list(), 
    # 事後分布を計算:式(2.48)
    density = MCMCpack::ddirichlet(x = phi_mat, alpha = alpha_k_lt[[1]]), # 確率密度
    fill_flg = !is.na(rowSums(phi_mat)), 
    density = dplyr::if_else(fill_flg, true = density, false = as.numeric(NA)), # 範囲外の値をNAに置換
    param = paste0(
      "n=", unique(n), " (", paste0(sum_s_k_lt[[1]], collapse = ", "), ")", 
      ", α=(", paste0(alpha_k_lt[[1]], collapse = ", "), ")"
    ) |> 
      factor(levels = posterior_level_vec[unique(n)+1]) # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup()


# レベルを設定用の文字列を作成
freq_vec <- rbind(rep(0, times = K), s_nk) |> 
  apply(MARGIN = 2, FUN = cumsum) |> # 累積和を計算
  apply(MARGIN = 1, FUN = paste0, collapse = ", ") # 試行ごとに結合
param_vec <- rbind(rep(0, times = K), s_nk) |> 
  apply(MARGIN = 2, FUN = cumsum) |> # 累積和を計算
  (\(.) {t(.) + alpha_k})() |> # 事後分布のパラメータを計算
  (\(.) {t(.) / colSums(.)})() |> # 予測分布のパラメータを計算
  round(digits = 2) |> 
  apply(MARGIN = 1, FUN = paste0, collapse = ", ") # 試行ごとに結合
predict_level_vec <- paste0("n=", 0:N, " (", freq_vec, "), π=(", param_vec, ")")

# 試行ごとに予測分布を格納
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  k = 1:K  # 次元番号
) |> # 試行ごとにx軸の値を複製
  dplyr::group_by(n) |> # 
  dplyr::mutate(
    # 予測分布のパラメータを計算:式(3.31')
    sum_s_k_lt = colSums(s_nk[0:unique(n), , drop = FALSE]) |> 
      list(), 
    alpha_k_lt = (colSums(s_nk[0:unique(n), , drop = FALSE]) + alpha_k) |> 
      list(), 
    pi_k_lt = (alpha_k_lt[[1]] / sum(alpha_k_lt[[1]])) |> 
      list(), 
    # 予測分布を格納
    prob = pi_k_lt[[1]], 
    param = paste0(
      "n=", unique(n), " (", paste0(sum_s_k_lt[[1]], collapse = ", "), ")", 
      ", π=(", paste0(round(pi_k_lt[[1]], digits = 2), collapse = ", "), ")"
    ) |> 
      factor(levels = predict_level_vec[unique(n)+1]) # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup()


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, s_nk[, 2] + 0.5 * s_nk[, 3]), # 三角座標への変換
  y = c(NA, sqrt(3) * 0.5 * s_nk[, 3]), # 三角座標への変換
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# 真のパラメータを格納
parameter_df <- tibble::tibble(
  x = pi_truth_k[2] + 0.5 * pi_truth_k[3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_truth_k[3], # 三角座標への変換
)

# 事後分布のアニメーションを作図
anime_posterior_graph <- ggplot() + 
  geom_segment(data = ternary_grid_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50", linetype = "dashed") + # 三角図のグリッド線
  geom_segment(data = ternary_axis_df, 
               mapping = aes(x = y_1_start, y = y_2_start, xend = y_1_end, yend = y_2_end), 
               color = "gray50") + # 三角図の枠線
  geom_text(data = ternary_ticklabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v, angle = angle)) + # 三角図の軸目盛ラベル
  geom_text(data = ternary_axislabel_df, 
            mapping = aes(x = y_1, y = y_2, label = label, hjust = h, vjust = v), 
            parse = TRUE, size = 6) + # 三角図の軸ラベル
  geom_contour_filled(data = anime_posterior_df, 
                      mapping = aes(x = y_1, y = y_2, z = density, fill = ..level..), 
                      alpha = 0.8) + # 事後分布
  geom_point(data = anime_data_df, mapping = aes(x = x, y = y, color = "data"), 
             size = 6) + # 観測データ
  geom_point(data = parameter_df, mapping = aes(x = x, y = y, color = "param"), 
             shape = 4, size = 6) + # 真のパラメータ
  gganimate::transition_manual(frames = param) + # フレーム
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = NULL) + # x軸
  scale_y_continuous(breaks = c(0, 0.25*sqrt(3), 0.5*sqrt(3)), labels = NULL) + # y軸
  scale_color_manual(breaks = c("param", "data"), 
                     values = c("red", "pink"), 
                     labels = c("true parameter", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(5, 5), shape = c(4, 19)))) + # 凡例の体裁:(凡例表示用)
  coord_fixed(ratio = 1, clip = "off") + # アスペクト比
  theme(axis.ticks = element_blank(), 
        panel.grid.minor = element_blank()) + # 図の体裁
  labs(title = "Dirichlet Distribution", 
       subtitle = "{current_frame}", 
       fill = "density", 
       x = "", y = "")

# gif画像を作成
gganimate::animate(
  plot = anime_posterior_graph, nframes = N+1+10, end_pause = 10, fps = 10, 
  width = 800, height = 800
)


# 観測データを格納
anime_data_df <- tibble::tibble(
  param = unique(anime_predict_df[["param"]]), # フレーム切替用ラベル
  k = c(NA, which(t(s_nk) == 1, arr.ind = TRUE)[, "row"]) # クラスタ番号
)

# 真の分布を複製
anime_model_df <- tidyr::expand_grid(
  k = 1:K, # 次元番号
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
) |> 
  dplyr::mutate(
    prob = pi_truth_k[k] # 確率
  )

# 予測分布を作図
anime_predict_graph <- ggplot() + 
  geom_bar(data = anime_predict_df, mapping = aes(x = k, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = anime_model_df, aes(x = k, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  geom_point(data = anime_data_df, aes(x = k, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(frames = param) + # フレーム
  scale_fill_manual(values = c(model = NA, predict = "purple", data = NA), na.value = NA, 
                    labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict = "purple", data = "pink"), 
                     labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = 1:K, minor_breaks = FALSE) + # x軸目盛
  guides(fill = guide_legend(override.aes = list(fill = c(NA, "purple", NA))), 
         color = guide_legend(override.aes = list(size = c(0.5, 0.5, 5), linetype = c("dashed", "blank", "blank"), shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(ylim = c(0, 1)) + # 表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = "{current_frame}", 
       x = "k", y = "probability")

# gif画像を作成
gganimate::animate(
  plot = anime_predict_graph, nframes = N+1+10, end_pause = 10, fps = 10, 
  width = 800, height = 800
)


