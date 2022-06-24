
# 3.4.3 多次元ガウス分布：平均・精度が未知の場合 ----------------------------------------------

# 3.4.3項で利用するパッケージ
library(tidyverse)
library(mvnfast)


### 尤度の設定 -----

# 真の平均パラメータを指定
mu_truth_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma2_truth_dd <- matrix(c(600, -100, -100, 400), nrow = 2, ncol = 2)
sqrt(sqrt(sigma2_truth_dd^2)) # (似非)相関行列を確認

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma2_truth_dd)


# 作図用のxのx軸の値を作成
x_1_vec <- seq(
  mu_truth_d[1] - 3 * sqrt(sigma2_truth_dd[1, 1]), 
  mu_truth_d[1] + 3 * sqrt(sigma2_truth_dd[1, 1]), 
  length.out = 500
)

# 作図用のxのy軸の値を作成
x_2_vec <- seq(
  mu_truth_d[2] - 3 * sqrt(sigma2_truth_dd[2, 2]), 
  mu_truth_d[2] + 3 * sqrt(sigma2_truth_dd[2, 2]), 
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
    X = x_point_mat, mu = mu_truth_d, sigma = solve(lambda_truth_dd)
  )
)

# 尤度を作図
ggplot(model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 尤度
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("mu=(", paste(round(mu_truth_d, 1), collapse = ", "), ")", 
                         ", lambda=(", paste(round(lambda_truth_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# 多次元ガウス分布に従うデータを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_truth_d, sigma = solve(lambda_truth_dd))
#x_nd <- mvtnorm::rmvnorm(n = N, mean = mu_truth_d, sigma = solve(lambda_truth_dd))

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
                         ", mu=(", paste(mu_truth_d, collapse = ", "), ")", 
                         ", lambda=(", paste(round(lambda_truth_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 事前分布(ガウス・ウィシャート分布)の設定 -----

# muの事前分布のパラメータを指定
m_d <- c(0, 0)
beta <- 1

# lambdaの事前分布のパラメータを指定
nu <- 2
w_dd <- matrix(c(0.0005, 0, 0, 0.0005), nrow = 2, ncol = 2)

# lambdaの期待値を計算:式(2.89)
E_lambda_dd <- nu * w_dd


# muの事前分布の分散共分散行列を計算
E_sigma2_mu_dd <- solve(beta * E_lambda_dd)

# 作図用のmuのx軸の値を作成
mu_1_vec <- seq(
  mu_truth_d[1] - 2 * sqrt(E_sigma2_mu_dd[1, 1]), 
  mu_truth_d[1] + 2 * sqrt(E_sigma2_mu_dd[1, 1]), 
  length.out = 500
)

# 作図用のmuのy軸の値を作成
mu_2_vec <- seq(
  mu_truth_d[2] - 2 * sqrt(E_sigma2_mu_dd[2, 2]), 
  mu_truth_d[2] + 2 * sqrt(E_sigma2_mu_dd[2, 2]), 
  length.out = 500
)

# 作図用のmuの点を作成
mu_point_mat <- cbind(
  rep(mu_1_vec, times = length(mu_2_vec)), 
  rep(mu_2_vec, each = length(mu_1_vec))
)

# 作図用に真のmuのデータフレームを作成
mu_df <- tibble(
  mu_1 = mu_truth_d[1], 
  mu_2 = mu_truth_d[2]
)


# muの事前分布を計算:式(2.72)
prior_mu_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = mu_point_mat, mu = m_d, sigma = solve(beta * E_lambda_dd)
  )
)

# muの事前分布を作図
ggplot() + 
  geom_contour(data = prior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事前分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("m=(", paste(round(m_d, 1), collapse = ", "), ")", 
                         ", beta=", beta, 
                         ", E[lambda]=(", paste(round(E_lambda_dd, 5), collapse = ", "), ")"), 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")


# muの期待値を計算:式(2.76)
E_mu_d <- m_d

# 事前分布の期待値を用いた分布を計算:式(2.72)
prior_E_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = E_mu_d, sigma = solve(E_lambda_dd)
  )
)

# 事前分布の期待値を用いた分布を作図
ggplot() + 
  geom_contour(data = prior_E_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 事前分布の期待値を用いた分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("m=(", paste(m_d, collapse = ", "), ")", 
                         ", nu=", nu, 
                         ", W=(", paste(round(w_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


### 事後分布(ガウス・ウィシャート分布)の計算 -----

# muの事後分布のパラメータを計算:式(3.129)
beta_hat <- N + beta
m_hat_d <- (colSums(x_nd) + beta * m_d) / beta_hat

# lambdaの事後分布のパラメータを計算:式(3.133)
term_x <- t(x_nd) %*% as.matrix(x_nd)
term_m <- beta * as.matrix(m_d) %*% t(m_d)
term_m_hat <- beta_hat * as.matrix(m_hat_d) %*% t(m_hat_d)
w_hat_dd <- solve(
  term_x + term_m - term_m_hat + solve(w_dd)
)
nu_hat <- N + nu


# lambdaの期待値を計算:式(2.89)
E_lambda_hat_dd <- nu_hat * w_hat_dd

# muの事後分布を計算:式(2.72)
posterior_mu_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = mu_point_mat, mu = m_hat_d, sigma = solve(beta_hat * E_lambda_hat_dd)
  )
)

# muの事後分布を作図
ggplot() + 
  geom_contour(data = posterior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事後分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  xlim(c(min(mu_point_mat[, 1]), max(mu_point_mat[, 1]))) + # x軸の表示範囲
  ylim(c(min(mu_point_mat[, 2]), max(mu_point_mat[, 2]))) + # y軸の表示範囲
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, 
                         ", m_hat=(", paste(round(m_hat_d, 1), collapse = ", "), ")", 
                         ", beta_hat=", beta_hat, 
                         ", E[lambda]=(", paste(round(E_lambda_hat_dd, 5), collapse = ", "), ")"), 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")


# muの期待値を計算:式(2.76)
E_mu_hat_d <- m_hat_d

# 事後分布の期待値を用いた分布を計算:式(2.72)
posterior_E_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = E_mu_hat_d, sigma = solve(E_lambda_hat_dd)
  )
)

# 事後分布の期待値を用いた分布を作図
ggplot() + 
  geom_contour(data = posterior_E_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 事後分布の期待値を用いた分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), 
             color = "red", shape = 4, size = 5) + # 真のmu
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = paste0("N=", N, 
                         ", m_hat=(", paste(round(m_hat_d, 1), collapse = ", "), ")", 
                         ", nu_hat=", nu_hat, 
                         ", W_hat=(", paste(round(w_hat_dd, 5), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


# 予測分布(多次元スチューデントのt分布)の計算 -----------------------------------------------------------------

# 次元数を取得
D <- length(m_d)

# 予測分布のパラメータを計算:式(3.140')
mu_s_hat_d <- m_hat_d
lambda_s_hat_dd <- (1 - D + nu_hat) * beta_hat / (1 + beta_hat) * w_hat_dd
nu_s_hat <- 1 - D + nu_hat


# 予測分布を計算:式(3.121)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvt(
    X = x_point_mat, mu = mu_s_hat_d, sigma = solve(lambda_s_hat_dd), df = nu_s_hat
  )
)

# 予測分布を作図
ggplot() + 
  geom_contour(data = predict_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 予測分布
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), 
             color = "red", shape = 4, size = 5) + # 真のmu
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = paste0("N=", N, 
                         ", mu_s_hat=(", paste(round(mu_s_hat_d, 1), collapse = ", "), ")", 
                         ", lambda_s_hat=(", paste(round(lambda_s_hat_dd, 5), collapse = ", "), ")", 
                         ", nu_s_hat=", nu_s_hat), 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")


# ・アニメーションによる推移の確認 -------------------------------------------------------------

# 3.4.3項で利用するパッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)


### モデルの設定 -----

# 真の平均パラメータを指定
mu_truth_d <- c(25, 50)

# 真の分散共分散行列を指定
sigma2_truth_dd <- matrix(c(600, 100, 100, 400), nrow = 2, ncol = 2)

# 真の精度行列を計算
lambda_truth_dd <- solve(sigma2_truth_dd)


# muの事前分布のパラメータを指定
m_d <- c(0, 0)
beta <- 1

# lambdaの事前分布のパラメータを指定
nu <- 2
w_dd <- matrix(c(0.0005, 0, 0, 0.0005), nrow = 2, ncol = 2)


# 初期値による予測分布のパラメータを計算:式(3.140)
mu_s_d <- m_d
lambda_s_dd <- (nu - 1) * beta / (1 + beta) * w_dd
nu_s <- nu - 1


# 作図用のmuのx軸の値を作成
mu_1_vec <- seq(
  mu_truth_d[1] - 1 * sqrt(solve(beta * nu * w_dd)[1, 1]), 
  mu_truth_d[1] + 1 * sqrt(solve(beta * nu * w_dd)[1, 1]), 
  length.out = 250
)

# 作図用のmuのy軸の値を作成
mu_2_vec <- seq(
  mu_truth_d[2] - 1 * sqrt(solve(beta * nu * w_dd)[2, 2]), 
  mu_truth_d[2] + 1 * sqrt(solve(beta * nu * w_dd)[2, 2]), 
  length.out = 250
)

# 作図用のmuの点を作成
mu_point_mat <- cbind(
  rep(mu_1_vec, times = length(mu_2_vec)), 
  rep(mu_2_vec, each = length(mu_1_vec))
)

# muの事前分布を計算:式(2.72)
posterior_mu_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = mu_point_mat, mu = m_d, sigma = solve(beta * nu * w_dd)
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", m=(", paste(round(m_d, 1), collapse = ", "), ")", 
      ", beta=", beta, 
      ", E[lambda]=(", paste0(round(nu * w_dd, 5), collapse = ", "), ")"
    )
  )
)


# 作図用のxのx軸の値を作成
x_1_vec <- seq(
  mu_truth_d[1] - 3 * sqrt(sigma2_truth_dd[1, 1]), 
  mu_truth_d[1] + 3 * sqrt(sigma2_truth_dd[1, 1]), 
  length.out = 250
)

# 作図用のxのy軸の値を作成
x_2_vec <- seq(
  mu_truth_d[2] - 3 * sqrt(sigma2_truth_dd[2, 2]), 
  mu_truth_d[2] + 3 * sqrt(sigma2_truth_dd[2, 2]), 
  length.out = 250
)

# 作図用のxの点を作成
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)

# 初期値による予測分布を計算:式(3.121)
predict_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvt(
    X = x_point_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
  ), 
  label = as.factor(
    paste0(
      "N=", 0, 
      ", mu_s=(", paste(round(mu_s_d, 1), collapse = ", "), ")", 
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
  x_nd[n, ] <- mvnfast::rmvn(n = 1, mu = mu_truth_d, sigma = solve(lambda_truth_dd)) %>% 
    as.vector()
  
  # muの事後分布のパラメータを更新:式(3.129)
  old_beta <- beta
  old_m_d <- m_d
  beta <- 1 + beta
  m_d <- (x_nd[n, ] + old_beta * m_d) / beta
  
  # lambdaの事後分布のパラメータを更新:式(3.133)
  term_x <- x_nd[n, ] %*% t(x_nd[n, ])
  term_m <- old_beta * old_m_d %*% t(old_m_d)
  term_m_hat <- beta * m_d %*% t(m_d)
  w_dd <- solve(
    term_x + term_m - term_m_hat + solve(w_dd)
  )
  nu <- 1 + nu
  
  # muの事前分布を計算:式(2.72)
  tmp_posterior_mu_df <- tibble(
    mu_1 = mu_point_mat[, 1], 
    mu_2 = mu_point_mat[, 2], 
    density = mvnfast::dmvn(
      X = mu_point_mat, mu = m_d, sigma = solve(beta * nu * w_dd)
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", m_hat=(", paste(round(m_d, 1), collapse = ", "), ")", 
        ", beta_hat=", beta, 
        ", E[lambda]=(", paste0(round(nu * w_dd, 5), collapse = ", "), ")"
      )
    )
  )
  
  # 予測分布のパラメータを更新:式(1.40)
  mu_s_d <- m_d
  lambda_s_dd <- (nu - 1) * beta / (1 + beta) * w_dd
  nu_s <- nu - 1
  
  # 初期値による予測分布を計算:式(3.121)
  tmp_predict_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = mvnfast::dmvt(
      X = x_point_mat, mu = mu_s_d, sigma = solve(lambda_s_dd), df = nu_s
    ), 
    label = as.factor(
      paste0(
        "N=", n, 
        ", mu_s_hat=(", paste(round(mu_s_d, 1), collapse = ", "), ")", 
        ", lambda_s_hat=(", paste(round(lambda_s_dd, 5), collapse = ", "), ")", 
        ", nu_s=", nu_s
      )
    )
  )
  
  # n回目の結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_posterior_mu_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
  
  # 動作確認
  print(paste0("n=", n, " (", round(n / N * 100, 1), "%)"))
}

# 観測データを確認
summary(x_nd)


### 作図処理 -----

# 作図用に真のmuのデータフレームを作成
mu_df <- tibble(
  mu_1 = mu_truth_d[1], 
  mu_2 = mu_truth_d[2]
)

# muの事後分布を作図
posterior_graph <- ggplot() + 
  geom_contour(data = posterior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # muの事後分布
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), color = "red", shape = 4, size = 5) + # 真のmu
  transition_manual(label) + # フレーム
  labs(title = "Multivariate Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu[1]), y = expression(mu[2]), 
       color = "density")

# gif画像を作成
animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.72)
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_d, sigma = solve(lambda_truth_dd)
  )
)

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
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 尤度
  geom_point(data = mu_df, aes(x = mu_1, y = mu_2), 
             color = "red", shape = 4, size = 5) + # 真のmu
  transition_manual(label) + # フレーム
  labs(title = "Multivariate Student's t Distribution", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]), 
       color = "density")

# gif画像を作成
animate(predict_graph, nframes = N + 1, fps = 10)

warnings()


