
# 3.4.2 多次元ガウス分布：精度が未知の場合 ----------------------------------------------

# 3.4.2項で利用するパッケージ
library(tidyverse)
library(mvnfast)


### 尤度(多次元ガウス分布)の設定 -----

# (既知の)平均パラメータを指定
mu_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma2_truth_dd <- matrix(c(600, -100, -100, 400), nrow = 2, ncol = 2)
sqrt(sqrt(sigma2_truth_dd^2)) # (似非)相関行列を確認

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma2_truth_dd)


# 作図用のxのx軸の値を作成
x_1_vec <- seq(
  mu_d[1] - 3 * sqrt(sigma2_truth_dd[1, 1]), 
  mu_d[1] + 3 * sqrt(sigma2_truth_dd[1, 1]), 
  length.out = 500
)

# 作図用のxのy軸の値を作成
x_2_vec <- seq(
  mu_d[2] - 3 * sqrt(sigma2_truth_dd[2, 2]), 
  mu_d[2] + 3 * sqrt(sigma2_truth_dd[2, 2]), 
  length.out = 500
)

# 作図用のxの点を作成
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)


# 尤度を計算:式(2.72)
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_d, sigma = solve(lambda_truth_dd)
  )
)

# 尤度を作図
ggplot(model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("mu=(", paste(round(mu_d, 1), collapse = ", "), ")", 
                         ", lambda=(", paste(round(lambda_truth_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_d, sigma = solve(lambda_truth_dd))
#x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_d, sigma = solve(lambda_truth_dd))

# 観測データを確認
summary(x_nd)


# 観測データのデータフレームを作成
x_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2]
)

# 観測データの散布図を作図
ggplot() + 
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, 
                         ", mu=(", paste(mu_d, collapse = ", "), ")", 
                         ", lambda=(", paste(round(lambda_truth_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 事前分布(ウィシャート分布)の設定 -----

# lambdaの事前分布のパラメータを指定
nu <- 2
w_dd <- matrix(c(0.0005, 0, 0, 0.0005), nrow = 2, ncol = 2)


# lambdaの期待値を計算:式(2.89)
E_lambda_dd <- nu * w_dd

# 事前分布の期待値を用いた分布を計算:式(2.72)
prior_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_d, sigma = solve(E_lambda_dd)
  )
)

# 事前分布の期待値を用いた分布を作図
ggplot() + 
  geom_contour(data = prior_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # lambdaの期待値を用いた分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("nu=", nu, 
                         ", W=(", paste(round(w_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 事後分布(ウィシャート分布)の計算 -----

# lambdaの事後分布のパラメータを計算:式(3.116)
w_hat_dd <- solve(
  (t(x_nd) - mu_d) %*% t(t(x_nd) - mu_d) + solve(w_dd)
)
nu_hat <- N + nu


# lambdaの期待値を計算:式(2.89)
E_lambda_hat_dd <- nu_hat * w_hat_dd

# 事後分布の期待値を用いた分布を計算:式(2.72)
posterior_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_d, sigma = solve(E_lambda_hat_dd)
  )
)

# 事後分布の期待値を用いた分布を作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # lambdaの期待値を用いた分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, 
                         ", nu_hat=", nu_hat, 
                         ", W_hat=(", paste(round(w_hat_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 予測分布(多次元スチューデントのt分布)の計算 -----

# 次元数を取得
D <- length(mu_d)

# 予測分布のパラメータを計算:式(3.124)
mu_s_d <- mu_d
lambda_s_hat_dd <- (1 - D + nu_hat) * w_hat_dd
nu_s_hat <- 1 - D + nu_hat


# 予測分布を計算:式(3.121)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvt(
    X = x_point_mat, mu = mu_s_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat
  )
)

# 予測分布を作図
ggplot() + 
  geom_contour(data = predict_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = paste0("N=", N, 
                         ", lambda_s_hat=(", paste(round(lambda_s_hat_dd, 5), collapse = ", "), ")", 
                         ", nu_s_hat=", nu_s_hat), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


# ・アニメーションによる推移の確認 ---------------------------------------------------------------------

# 3.4.2項で利用するパッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)


### モデルの設定 -----

# (既知の)平均パラメータを指定
mu_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma2_truth_dd <- matrix(c(600, 100, 100, 400), nrow = 2, ncol = 2)

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma2_truth_dd)


# lambdaの事前分布のパラメータを指定
nu <- 2
w_dd <- matrix(c(0.0005, 0, 0, 0.0005), nrow = 2, ncol = 2)


# 初期値による予測分布のパラメータを計算:式(3.124)
mu_s_d <- mu_d
lambda_s_dd <- (nu - 1) * w_dd
nu_s <- nu - 1


# 作図用のxのx軸の値を作成
x_1_vec <- seq(
  mu_d[1] - 3 * sqrt(sigma2_truth_dd[1, 1]), 
  mu_d[1] + 3 * sqrt(sigma2_truth_dd[1, 1]), 
  length.out = 250
)

# 作図用のxのy軸の値を作成
x_2_vec <- seq(
  mu_d[2] - 3 * sqrt(sigma2_truth_dd[2, 2]), 
  mu_d[2] + 3 * sqrt(sigma2_truth_dd[2, 2]), 
  length.out = 250
)

# 作図用のxの点を作成
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)


# 事前分布の期待値を用いた分布を計算:式(2.72)
posterior_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_d, sigma = solve(nu * w_dd)
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", nu=", nu, 
      ", W=(", paste(round(w_dd, 5), collapse = ", "), ")"
    )
  )
)

# 予測分布を計算:式(3.121)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvt(
    X = x_point_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", lambda_s=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")", 
      ", nu_s=", nu_s
    )
  )
)


### 推論処理 -----

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を初期化
x_nd <- matrix(0, nrow = N, ncol = 2)

# ベイズ推論
for(n in 1:N) {
  
  # 多次元ガウス分布に従うデータを生成
  x_nd[n, ] <- mvnfast::rmvn(n = 1, mu = mu_d, sigma = solve(lambda_truth_dd)) %>% 
    as.vector()
  
  # 事後分布のパラメータを更新:式(3.116)
  w_dd <- solve(
    (x_nd[n, ] - mu_d) %*% t(x_nd[n, ] - mu_d) + solve(w_dd)
  )
  nu <- 1 + nu
  
  # 事後分布の期待値を用いた分布を計算:式(2.72)
  tmp_posterior_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = mvnfast::dmvn(
      X = x_point_mat, mu = mu_d, sigma = solve(nu * w_dd)
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", nu_hat=", nu, 
        ", W_hat=(", paste(round(w_dd, 5), collapse = ", "), ")"
      )
    )
  )
  
  # 予測分布のパラメータを更新:式(3.124)
  #mu_s_d <- mu_d
  lambda_s_dd <- (nu - 1) * w_dd
  nu_s <- nu - 1
  
  # 予測分布を計算:式(3.121)
  tmp_predict_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = mvnfast::dmvt(
      X = x_point_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", lambda_s_hat=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")", 
        ", nu_s_hat=", nu_s
      )
    )
  )
  
  # n回目の結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(paste0("n=", n, " (", round(n / N * 100, 1), "%)"))
}

# 観測データを確認
summary(x_nd)


### 作図処理 -----

# 尤度を計算:式(2.72)
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_d, sigma = solve(lambda_truth_dd)
  )
)


# 作図用に観測データのデータフレームを作成
label_vec <- unique(posterior_df[["label"]]) # lambdaの期待値を用いた分布のラベルを抽出
x_df <- tibble(x_n1 = NA, x_n2 = NA, label = label_vec[1]) # 初期値用
for(n in 1:N) {
  # n個目までのデータフレームを作成
  tmp_x_df <- tibble(
    x_n1 = x_nd[1:n, 1], 
    x_n2 = x_nd[1:n, 2], 
    label = label_vec[n + 1]
  )
  
  # 結合
  x_df <- rbind(x_df, tmp_x_df)
}

# 事後分布の期待値を用いた分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # lambdaの期待値を用いた分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  transition_manual(label) + # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")

# gif画像を作成
animate(posterior_graph, nframes = N + 1, fps = 10)


# 作図用に観測データのデータフレームを作成
label_vec <- unique(predict_df[["label"]]) # 予測分布のラベルを抽出
x_df <- tibble(x_n1 = NA, x_n2 = NA, label = label_vec[1]) # 初期値用
for(n in 1:N) {
  # n個目までのデータフレームを作成
  tmp_x_df <- tibble(
    x_n1 = x_nd[1:n, 1], 
    x_n2 = x_nd[1:n, 2], 
    label = label_vec[n + 1]
  )
  
  # 結合
  x_df <- rbind(x_df, tmp_x_df)
}

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour(data = predict_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  transition_manual(label) + # フレーム
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")

# gif画像を作成
animate(predict_graph, nframes = N + 1, fps = 10)


