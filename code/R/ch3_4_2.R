
# 3.4.2 多次元ガウス分布：精度が未知の場合 ----------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)

# チェック用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(多次元ガウス分布)の設定 -----

# 次元数を指定
D <- 2

# (既知の)平均ベクトルを指定
mu_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma_truth_dd <- matrix(c(900, -100, -100, 400), nrow = D, ncol = D)

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma_truth_dd)


# グラフ用のxの値を作成
x_1_vec <- seq(
  mu_d[1] - sqrt(sigma_truth_dd[1, 1]) * 3, 
  mu_d[1] + sqrt(sigma_truth_dd[1, 1]) * 3, 
  length.out = 201
)
x_2_vec <- seq(
  mu_d[2] - sqrt(sigma_truth_dd[2, 2]) * 3, 
  mu_d[2] + sqrt(sigma_truth_dd[2, 2]) * 3, 
  length.out = 201
)

# グラフ用のxの点を作成
x_mat <- tidyr::expand_grid(
  x_1 = x_1_vec, 
  x_2 = x_2_vec
) |> # 格子点を作成
  as.matrix() # マトリクスに変換


# 真の分布(多次元ガウス分布)を計算:式(2.72)
model_dens_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = x_mat, mu = mu_d, sigma = sigma_truth_dd) # 確率密度
)

# パラメータラベル用の文字列を作成
model_param_text <- paste0(
  "list(mu==(list(", paste(round(mu_d, 1), collapse = ", "), "))", 
  ", Lambda==(list(", paste(round(lambda_truth_dd, 5), collapse = ", "), ")))"
)

# 真の分布を作図
ggplot() + 
  #geom_contour(data = model_dens_df, mapping = aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 真の分布:(等高線図)
  geom_contour_filled(data = model_dens_df, mapping = aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 真の分布:(塗りつぶし等高線図)
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = model_param_text), 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))


# 真のΛによるマハラノビス距離を計算
model_delta_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  delta = mahalanobis(x = x_mat, center = mu_d, cov = sigma_truth_dd) |> 
    sqrt() # 距離
)

# 真のΛの固有値・固有ベクトルを計算
model_eigen <- eigen(sigma_truth_dd)
model_lmd_d <- model_eigen[["values"]]
model_u_dd  <- model_eigen[["vectors"]] |> 
  t()

# 真のΛによる楕円の軸を計算
model_axis_df <- tibble::tibble(
  xend = mu_d[1] + model_u_dd[, 1] * sqrt(model_lmd_d), # x軸の値
  yend = mu_d[2] + model_u_dd[, 2] * sqrt(model_lmd_d)  # y軸の値
)

# 等高線を引く値を指定
break_vals <- seq(0, ceiling(max(model_delta_df[["delta"]])), by = 0.5)

# 真の分散共分散行列による距離と軸を作図
ggplot() + 
  #geom_contour(data = model_delta_df, aes(x = x_1, y = x_2, z = delta, color = ..level..), 
  #             breaks = break_vals) + # 真のΛによる距離:(等高線図)
  geom_contour_filled(data = model_delta_df, aes(x = x_1, y = x_2, z = delta, fill = ..level..), 
                      breaks = break_vals, alpha = 0.8) + # 真のΛによる距離:(塗りつぶし等高線図)
  geom_segment(data = model_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "model"), 
               color = "red", size = 1, arrow = arrow(length = unit(10, "pt"))) + # 真のΛによる軸
  scale_linetype_manual(breaks = "model", values = "solid", labels = "true model", name = "axis") + # (凡例表示用の黒魔術)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Mahalanobis Distance", 
       subtitle = parse(text = model_param_text), 
       color = "distance", fill = "distance", 
       x = expression(x[1]), y = expression(x[2]))


### ・データの生成 -----

# (観測)データ数を指定
N <- 100


# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_d, sigma = sigma_truth_dd)

# 観測データを格納
data_df <- tibble::tibble(
  x_1 = x_nd[, 1], # x軸の値
  x_2 = x_nd[, 2]  # y軸の値
)

# パラメータラベル用の文字列を作成
sample_param_text <- paste0(
  "list(mu==(list(", paste(round(mu_d, 1), collapse = ", "), "))", 
  ", Lambda==(list(", paste(round(lambda_truth_dd, 5), collapse = ", "), "))", 
  ", N==", N, ")"
)

# 観測データの散布図を作成
ggplot() + 
  #geom_contour(data = model_dens_df, aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 真の分布:(等高線図)
  geom_contour_filled(data = model_dens_df, aes(x = x_1, y = x_2, z = dens, fill = ..level..), alpha = 0.8) + # 真の分布:(塗りつぶし等高線図)
  geom_point(data = data_df, aes(x = x_1, y = x_2), color = "orange") + # 観測データ
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = parse(text = sample_param_text), 
       color = "density", fill = "density", 
       x = expression(x[1]), y = expression(x[2]))


### ・事前分布(ウィシャート分布)の設定 -----

# Λの事前分布の自由度を指定
nu <- D

# Λの事前分布の逆スケール行列を指定
w_dd <- diag(D) * 0.0001


# Λの事前分布の期待値を計算:式(2.89)
E_lambda_dd <- nu * w_dd

# Λの期待値によるマハラノビス距離を計算
prior_delta_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  delta = mahalanobis(x = x_mat, center = mu_d, cov = E_lambda_dd, inverted = TRUE) |> 
    sqrt() # 距離
)

# Λの期待値の固有値・固有ベクトルを計算
prior_eigen <- eigen(solve(E_lambda_dd))
prior_lmd_d <- prior_eigen[["values"]]
prior_u_dd  <- prior_eigen[["vectors"]] |> 
  t()

# Λの期待値による楕円の軸を計算
prior_axis_df <- tibble::tibble(
  xend = mu_d[1] + prior_u_dd[, 1] * sqrt(prior_lmd_d), # x軸の値
  yend = mu_d[2] + prior_u_dd[, 2] * sqrt(prior_lmd_d)  # y軸の値
)


# パラメータラベル用の文字列を作成
prior_param_text <- paste0(
  "list(nu==", nu, 
  ", W==(list(", paste(w_dd, collapse = ", "), ")))"
)

# Λの事前分布の期待値による距離と軸を作図
ggplot() + 
  geom_contour(data = model_delta_df, mapping = aes(x = x_1, y = x_2, z = delta, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真のΛによる距離
  geom_contour(data = prior_delta_df, mapping = aes(x = x_1, y = x_2, z = delta, color = ..level..)) + # Λの期待値による距離
  geom_segment(data = model_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "model"), 
               color = "red", size = 1, arrow = arrow(length = unit(10, "pt"))) + # 真のΛによる軸
  geom_segment(data = prior_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "prior"), 
               color = "blue", size = 1, arrow = arrow(length = unit(10, "pt"))) + # Λの期待値による軸
  scale_linetype_manual(breaks = c("prior", "model"), 
                        values = c("solid", "dashed"), 
                        labels = c("prior", "true model"), name = "axis") + # (凡例表示用の黒魔術)
  guides(linetype = guide_legend(override.aes = list(color = c("blue", "red"), 
                                                     size = c(0.5, 0.5), 
                                                     linetype = c("solid", "dashed")))) + # (凡例表示用の黒魔術)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Mahalanobis Distance", 
       subtitle = parse(text = prior_param_text), 
       color = "distance", 
       x = expression(x[1]), y = expression(x[2]))


### ・事後分布(ウィシャート分布)の計算 -----

# Λの事後分布の自由度を計算:式(3.116)
nu_hat <- N + nu

# Λの事後分布の逆スケール行列を計算:式(3.116)
w_hat_dd <- solve((t(x_nd) - mu_d) %*% t(t(x_nd) - mu_d) + solve(w_dd))


# Λの事後分布の期待値を計算:式(2.89)
E_lambda_hat_dd <- nu_hat * w_hat_dd

# Λの期待値によるマハラノビス距離を計算
posterior_delta_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  delta = mahalanobis(x = x_mat, center = mu_d, cov = E_lambda_hat_dd, inverted = TRUE) |> 
    sqrt() # 距離
)

# Λの期待値の固有値・固有ベクトルを計算
posterior_eigen <- eigen(solve(E_lambda_hat_dd))
posterior_lmd_d <- posterior_eigen[["values"]]
posterior_u_dd  <- posterior_eigen[["vectors"]] |> 
  t()

# Λの期待値による楕円の軸を計算
posterior_axis_df <- tibble::tibble(
  xend = mu_d[1] + posterior_u_dd[, 1] * sqrt(posterior_lmd_d), # x軸の値
  yend = mu_d[2] + posterior_u_dd[, 2] * sqrt(posterior_lmd_d)  # y軸の値
)

# パラメータラベル用の文字列を作成
posterior_param_text <- paste0(
  "list(N==", N, 
  ", hat(nu)==", nu_hat, 
  ", hat(W)==(list(", paste(w_hat_dd, collapse = ", "), ")))"
)

# Λの事後分布の期待値による距離と軸を作図
ggplot() + 
  geom_contour(data = model_delta_df, aes(x = x_1, y = x_2, z = delta, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真のΛによる距離
  geom_contour(data = posterior_delta_df, aes(x = x_1, y = x_2, z = delta, color = ..level..)) + # Λの期待値による距離
  geom_segment(data = model_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "model"), 
               color = "red", size = 1, arrow = arrow(length = unit(10, "pt"))) + # 真のΛによる軸
  geom_segment(data = posterior_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "posterior"), 
               color = "blue", size = 1, arrow = arrow(length = unit(10, "pt"))) + # Λの期待値による軸
  scale_linetype_manual(breaks = c("posterior", "model"), 
                        values = c("solid", "dashed"), 
                        labels = c("posterior", "true model"), name = "axis") + # (凡例表示用の黒魔術)
  guides(linetype = guide_legend(override.aes = list(color = c("blue", "red"), 
                                                     size = c(0.5, 0.5), 
                                                     linetype = c("solid", "dashed")))) + # (凡例表示用の黒魔術)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Mahalanobis Distance", 
       subtitle = parse(text = posterior_param_text), 
       color = "distance", 
       x = expression(x[1]), y = expression(x[2]))


### ・予測分布(多次元スチューデントのt分布)の計算 -----

# 予測分布の位置ベクトルを計算:式(3.124')
mu_s_d <- mu_d

# 予測分布の逆スケール行列を計算:式(3.124')
lambda_s_hat_dd <- (1 - D + nu_hat) * w_hat_dd

# 予測分布の自由度を計算:式(3.124')
nu_s_hat <- 1 - D + nu_hat


# 予測分布(多次元t分布)を計算:式(3.121)
predict_dens_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvt(X = x_mat, mu = mu_s_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat) # 確率密度
)

# パラメータラベル用の文字列を作成
predict_param_text <- paste0(
  "list(N==", N, 
  ", mu[s]==(list(", paste0(mu_s_d, collapse = ", "), "))", 
  ", hat(Lambda)[s]==(list(", paste(round(lambda_s_hat_dd, 5), collapse = ", "), "))", 
  ", hat(nu)[s]==", nu_s_hat, ")"
)

# 予測分布を作図
ggplot() + 
  geom_contour(data = model_dens_df, aes(x = x_1, y = x_2, z = dens, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_contour(data = predict_dens_df, aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 予測分布:(等高線図)
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = parse(text = predict_param_text), 
       color = "density", 
       x = expression(x[1]), y = expression(x[2]))


# アニメーションによる学習推移の可視化 ----------------------------------------------------

### ・モデルの設定 -----

# 次元数を指定
D <- 2

# (既知の)平均ベクトルを指定
mu_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma_truth_dd <- matrix(c(900, -100, -100, 400), nrow = D, ncol = D)

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma_truth_dd)

# Λの事前分布の自由度を指定
nu <- D

# Λの事前分布の逆スケール行列を指定
w_dd <- diag(D) * 0.00005

# データ数(試行回数)を指定
N <- 100


# グラフ用のxの値を作成
x_1_vec <- seq(
  mu_d[1] - sqrt(sigma_truth_dd[1, 1]) * 3, 
  mu_d[1] + sqrt(sigma_truth_dd[1, 1]) * 3, 
  length.out = 201
)
x_2_vec <- seq(
  mu_d[2] - sqrt(sigma_truth_dd[2, 2]) * 3, 
  mu_d[2] + sqrt(sigma_truth_dd[2, 2]) * 3, 
  length.out = 201
)

# グラフ用のxの点を作成
x_mat <- tidyr::expand_grid(x_1 = x_1_vec, x_2 = x_2_vec) |> # 格子点を作成
  as.matrix() # マトリクスに変換


### ・推論処理：for関数による処理 -----

# Λの期待値を計算:式(2.89)
E_lambda_dd <- nu * w_dd

# Λの期待値によるマハラノビス距離を計算
anime_posterior_delta_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  delta = mahalanobis(x = x_mat, center = mu_d, cov = E_lambda_dd, inverted = TRUE) |> 
    sqrt(), # 距離
  param = paste0(
    "N=", 0, 
    ", nu=", nu, 
    ", W=(", paste0(round(w_dd, 6), collapse = ", "), ")"
  ) |> 
    as.factor() # フレーム切替用ラベル
)

# Λの期待値の固有値・固有ベクトルを計算
prior_eigen <- eigen(solve(E_lambda_dd))
prior_lmd_d <- prior_eigen[["values"]]
prior_u_dd  <- prior_eigen[["vectors"]] |> 
  t()

# Λの期待値による楕円の軸を計算
anime_posterior_axis_df <- tibble::tibble(
  xend = mu_d[1] + prior_u_dd[, 1] * sqrt(prior_lmd_d), 
  yend = mu_d[2] + prior_u_dd[, 2] * sqrt(prior_lmd_d), 
  param = paste0(
    "N=", 0, 
    ", nu=", nu, 
    ", W=(", paste0(round(w_dd, 6), collapse = ", "), ")"
  ) |> 
    as.factor() # フレーム切替用ラベル
)


# 初期値による予測分布のパラメータを計算:式(3.124)
mu_s_d      <- mu_d
lambda_s_dd <- (nu - 1) * w_dd
nu_s        <- nu - 1

# 予測分布(多次元t分布)を計算:式(3.121)
anime_predict_dens_df <- tibble::tibble(
  x_1 = x_mat[, 1], 
  x_2 = x_mat[, 2], 
  dens = mvnfast::dmvt(X = x_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s), # 確率密度
  param = paste0(
    "N=", 0, 
    ", mu_s=(", paste0(mu_s_d, collapse = ", "), ")", 
    ", lambda_s=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")", 
    ", nu_s=", nu_s
  ) |> 
  as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を初期化
x_nd <- matrix(NA, nrow = N, ncol = D)

# ベイズ推論
for(n in 1:N) {
  
  # 多次元ガウス分布に従うデータを生成
  x_nd[n, ] <- mvnfast::rmvn(n = 1, mu = mu_d, sigma = sigma_truth_dd) |> 
    as.vector()
  
  # Λの事後分布のパラメータを更新:式(3.116)
  nu   <- 1 + nu
  w_dd <- solve((x_nd[n, ] - mu_d) %*% t(x_nd[n, ] - mu_d) + solve(w_dd))
  
  # Λの期待値を計算:式(2.89)
  E_lambda_dd <- nu * w_dd
  
  # Λの期待値によるマハラノビス距離を計算
  tmp_posterior_delta_df <- tibble::tibble(
    x_1 = x_mat[, 1], 
    x_2 = x_mat[, 2], 
    delta = mahalanobis(x = x_mat, center = mu_d, cov = E_lambda_dd, inverted = TRUE) |> 
      sqrt(), # 距離
    param = paste0(
      "N=", n, 
      ", nu=", nu, 
      ", W=(", paste0(round(w_dd, 6), collapse = ", "), ")"
    ) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # Λの期待値の固有値・固有ベクトルを計算
  posterior_eigen <- eigen(solve(E_lambda_dd))
  posterior_lmd_d <- posterior_eigen[["values"]]
  posterior_u_dd  <- posterior_eigen[["vectors"]] |> 
    t()
  
  # Λの期待値による楕円の軸を計算
  tmp_posterior_axis_df <- tibble::tibble(
    xend = mu_d[1] + posterior_u_dd[, 1] * sqrt(posterior_lmd_d), 
    yend = mu_d[2] + posterior_u_dd[, 2] * sqrt(posterior_lmd_d), 
    param = paste0(
      "N=", n, 
      ", nu=", nu, 
      ", W=(", paste0(round(w_dd, 6), collapse = ", "), ")"
    ) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.124)
  #mu_s_d     <- mu_d
  lambda_s_dd <- (nu - 1) * w_dd
  nu_s        <- nu - 1
  
  # 予測分布(多次元t分布)を計算:式(3.121)
  tmp_predict_dens_df <- tibble::tibble(
    x_1 = x_mat[, 1], 
    x_2 = x_mat[, 2], 
    dens = mvnfast::dmvt(X = x_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s), 
    param = paste0(
        "N=", n, 
        ", mu_s=(", paste0(mu_s_d, collapse = ", "), ")", 
        ", lambda_s=(", paste0(round(lambda_s_dd, 5), collapse = ", "), ")", 
        ", nu_s=", nu_s
      ) |> 
    as.factor() # フレーム切替用ラベル
  )
  
  # 推論結果を結合
  anime_posterior_delta_df <- rbind(anime_posterior_delta_df, tmp_posterior_delta_df)
  anime_posterior_axis_df  <- rbind(anime_posterior_axis_df, tmp_posterior_axis_df)
  anime_predict_dens_df    <- rbind(anime_predict_dens_df, tmp_predict_dens_df)
  
  # 動作確認
  print(paste0("n=", n, " (", round(n / N * 100, 1), "%)"))
}

# 観測データを確認
summary(x_nd)


### ・推論処理：tidyverseパッケージによる処理 -----

# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_d, sigma = sigma_truth_dd)

# 試行ごとにΛの事後分布のパラメータを計算
anime_posterior_param_df <- tibble::tibble(
  n = 0:N, # データ番号(試行回数)
  nu_hat = n + nu
) |> 
  dplyr::group_by(n) |> # パラメータの計算用にグループ化
  dplyr::mutate(
    tmp_x_lt = dplyr::if_else(
      n > 0, # 事前分布を除く
      true = t(t(x_nd) - mu_d)[0:n, , drop = FALSE] |> 
        list(), # リストに格納
      false = matrix(0, nrow = 1, ncol = D) |> 
        list() # リストに格納
    ), 
    w_hat_lt = dplyr::if_else(
      n > 0, # 事前分布を除く
      true = solve(t(tmp_x_lt[[1]]) %*% tmp_x_lt[[1]] + solve(w_dd)) |> 
        list(), # リストに格納
      false = w_dd |> 
        list() # リストに格納
    )
  ) |> 
  dplyr::ungroup() # グループ化を解除

# 試行ごとにΛの期待値によるマハラノビス距離を計算
anime_posterior_delta_df <- tidyr::expand_grid(
  n = 0:N, # データ番号(試行回数)
  x_1 = x_1_vec, 
  x_2 = x_2_vec, 
) |> # 試行ごとに格子点を複製
  dplyr::group_by(n) |> # 距離の計算用にグループ化
  dplyr::mutate(
    delta = mahalanobis(
      x = x_mat, 
      center = mu_d, 
      cov = anime_posterior_param_df[["nu_hat"]][unique(n)+1] * anime_posterior_param_df[["w_hat_lt"]][[unique(n)+1]], # Λの期待値を計算:式(2.89)
      inverted = TRUE
    ) |> 
      sqrt(), # 距離
    param = paste0(
      "N=", unique(n), 
      ", nu=", anime_posterior_param_df[["nu_hat"]][unique(n)+1], 
      ", W=(", paste0(round(anime_posterior_param_df[["w_hat_lt"]][[unique(n)+1]], 6), collapse = ", "), ")"
    ) |> 
      as.factor() # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup()

# 試行ごとにΛの期待値による楕円の軸を計算
anime_posterior_axis_df <- tidyr::expand_grid(
  n = 0:N, # データ番号(試行回数)
  name = c("y_1", "y_2") # 値の受け皿
) |> # 試行ごとに受け皿を複製
  dplyr::group_by(n) |> # 軸の計算用にグループ化
  dplyr::mutate(
    # Λの期待値の固有値・固有ベクトルを計算
    eigen_lt = (anime_posterior_param_df[["nu_hat"]][unique(n)+1] * anime_posterior_param_df[["w_hat_lt"]][[unique(n)+1]]) |> # Λの期待値を計算:式(2.89)
      solve() |> # 分散共分散行列に変換
      eigen() |> 
      list(), # リストに格納
    
    # Λの期待値による楕円の軸を計算
    xend = mu_d[1] + eigen_lt[[1]][["vectors"]][1, ] * sqrt(eigen_lt[[1]][["values"]]), # x軸の値
    yend = mu_d[2] + eigen_lt[[1]][["vectors"]][2, ] * sqrt(eigen_lt[[1]][["values"]]), # y軸の値
    param = paste0(
      "N=", unique(n), 
      ", nu=", anime_posterior_param_df[["nu_hat"]][unique(n)+1], 
      ", W=(", paste0(round(anime_posterior_param_df[["w_hat_lt"]][[unique(n)+1]], 6), collapse = ", "), ")"
    ) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup() # グループ化を解除

# 試行ごとに予測分布を計算
anime_predict_dens_df <- tidyr::expand_grid(
  n = 0:N, # データ番号(試行回数)
  x_1 = x_1_vec, 
  x_2 = x_2_vec
) |> # 試行ごとに格子点を複製
  dplyr::group_by(n) |> # 分布の計算用にグループ化
  dplyr::mutate(
    # 予測分布のパラメータを計算:式(3.124')
    lambda_s_lt = ((1 - D + unique(n) + nu) * anime_posterior_param_df[["w_hat_lt"]][[unique(n)+1]]) |> 
        list(), # リストに格納
    nu_s = 1 - D + unique(n) + nu, 
    
    # 予測分布(多次元t分布)を計算:式(2.72)
    dens = mvnfast::dmvt(
      X = x_mat, 
      mu = mu_d, 
      sigma = solve(lambda_s_lt[[1]]), 
      df = unique(nu_s)
    ), # 確率密度
    param = paste0(
      "N=", unique(n), 
      ", mu_s=(", paste0(mu_d, collapse = ", "), ")", 
      ", lambda_s=(", paste0(round(lambda_s_lt[[1]], 5), collapse = ", "), ")", 
      ", nu_s=", nu_s
    ) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) |> 
  dplyr::ungroup() # グループ化を解除


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x_1 = c(NA, x_nd[, 1]), # x軸の値
  x_2 = c(NA, x_nd[, 2]), # y軸の値
  param = anime_posterior_delta_df[["param"]] |> 
    unique() # フレーム切替用ラベル
)

# 真のΛによるマハラノビス距離を計算
model_delta_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  delta = mahalanobis(x = x_mat, center = mu_d, cov = lambda_truth_dd, inverted = TRUE) |> 
    sqrt() # 距離
)

# 真のΛの固有値・固有ベクトルを計算
model_eigen <- eigen(sigma_truth_dd)
model_lmd_d <- model_eigen[["values"]]
model_u_dd  <- model_eigen[["vectors"]] |> 
  t()

# 真のΛによる楕円の軸を計算
model_axis_df <- tibble::tibble(
  xend = mu_d[1] + model_u_dd[, 1] * sqrt(model_lmd_d), # x軸の値
  yend = mu_d[2] + model_u_dd[, 2] * sqrt(model_lmd_d)  # y軸の値
)

# 等高線を引く値を指定
break_vals <- seq(0, ceiling(max(model_delta_df[["delta"]])), by = 0.5)

# Λの事後分布の期待値による距離と軸のアニメーションを作図
anime_posterior_graph <- ggplot() + 
  geom_contour(data = model_delta_df, aes(x = x_1, y = x_2, z = delta, color = ..level..), 
               breaks = break_vals, alpha = 0.5, linetype = "dashed") + # 真のΛによる距離
  geom_contour(data = anime_posterior_delta_df, aes(x = x_1, y = x_2, z = delta, color = ..level..), 
               breaks = break_vals) + # Λの期待値による距離
  geom_segment(data = model_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "model"), 
               color = "red", size = 1, arrow = arrow(length = unit(10, "pt"))) + # 真のΛによる軸
  geom_segment(data = anime_posterior_axis_df, mapping = aes(x = mu_d[1], y = mu_d[2], xend = xend, yend = yend, linetype = "posterior"), 
               color = "blue", size = 1, arrow = arrow(length = unit(10, "pt"))) + # Λの期待値による軸
  geom_point(data = anime_data_df, mapping = aes(x = x_1, y = x_2), 
             color = "orange", size = 3) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_linetype_manual(breaks = c("posterior", "model"), 
                        values = c("solid", "dashed"), 
                        labels = c("posterior", "true model"), name = "axis") + # (凡例表示用の黒魔術)
  guides(linetype = guide_legend(override.aes = list(color = c("blue", "red"), 
                                                     size = c(0.5, 0.5), 
                                                     linetype = c("solid", "dashed")))) + # (凡例表示用の黒魔術)
  coord_fixed(ratio = 1, xlim = c(min(x_1_vec), max(x_1_vec)), ylim = c(min(x_2_vec), max(x_2_vec))) + # アスペクト比
  labs(title = "Mahalanobis Distance", 
       subtitle = "{current_frame}", 
       color = "distance", 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(anime_posterior_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


# 観測データを格納
anime_data_df <- tibble::tibble(
  x_1 = c(NA, x_nd[, 1]), # x軸の値
  x_2 = c(NA, x_nd[, 2]), # y軸の値
  param = anime_predict_dens_df[["param"]] |> 
    unique() # フレーム切替用ラベル
)

# 観測データを複製して格納
anime_alldata_df <- tidyr::expand_grid(
  frame = 1:N, # フレーム番号
  n = 1:N # 試行回数
) |> # 全ての組み合わせを作成
  dplyr::filter(n < frame) |> # フレーム番号以前のデータを抽出
  dplyr::mutate(
    x_1 = x_nd[n, 1], # x軸の値
    x_2 = x_nd[n, 2], # y軸の値
    param = unique(anime_predict_dens_df[["param"]])[frame+1] # フレーム切替用ラベル
  )

# 真の分布(多次元ガウス分布)を計算:式(2.72)
model_dens_df <- tibble::tibble(
  x_1 = x_mat[, 1], # x軸の値
  x_2 = x_mat[, 2], # y軸の値
  dens = mvnfast::dmvn(X = x_mat, mu = mu_d, sigma = sigma_truth_dd) # 確率密度
)

# 予測分布を作図
anime_predict_graph <- ggplot() + 
  geom_contour(data = anime_predict_dens_df, aes(x = x_1, y = x_2, z = dens, color = ..level..)) + # 予測分布
  geom_contour(data = model_dens_df, aes(x = x_1, y = x_2, z = dens, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = anime_alldata_df, aes(x = x_1, y = x_2), 
             color  ="orange", alpha = 0.5) + # 過去の観測データ
  geom_point(data = anime_data_df, aes(x = x_1, y = x_2), 
             color  ="orange", size = 3) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = "{current_frame}", 
       color = "density", 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(anime_predict_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


