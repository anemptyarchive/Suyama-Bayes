
# 3.3.3 1次元ガウス分布の学習と予測：平均・精度が未知の場合 --------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)
library(LaplacesDemon)

# チェック用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(ガウス分布)の設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01
sqrt(1 / lambda_truth) # 標準偏差


# グラフ用のxの値を作成
x_vec <- seq(
  mu_truth - 1/sqrt(lambda_truth) * 4, 
  mu_truth + 1/sqrt(lambda_truth) * 4, 
  length.out = 201
)

# 真の分布を計算:式(2.64)
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_truth, sd = 1/sqrt(lambda_truth)) # 確率密度
)

# 真の分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1) + # 真の分布
  scale_color_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # 線の色:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda_truth, ")")), 
       x = "x", y = "density")


### ・データの生成 -----

# (観測)データ数を指定
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda_truth))

# 観測データを格納
data_df <- tibble::tibble(x = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_histogram(data = data_df, mapping = aes(x = x, y = ..density.., fill = "data"), 
                 bins = 30) + # 観測データ:(密度)
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "blank")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda_truth, ", N==", N, ")")), 
       x = "x", y = "density")


### ・事前分布(ガウス・ガンマ分布)の設定 -----

# μの事前分布のパラメータを指定
m    <- 0
beta <- 1

# λの事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のμの値を作成
mu_vec <- seq(mu_truth - 40, mu_truth + 40, length.out = 201)

# グラフ用のλの値を作成
lambda_vec <- seq(0, lambda_truth * 4, length.out = 201)

# グラフ用のμとλの点を作成
mu_lambda_mat <- tidyr::expand_grid(mu = mu_vec, lambda = lambda_vec) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# μとλの(同時)事前分布を計算
prior_df <- tidyr::tibble(
  mu = mu_lambda_mat[, 1], # 確率変数μ
  lambda = mu_lambda_mat[, 2], # 確率変数λ
  N_dens = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)), # μの確率密度
  Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
  density = N_dens * Gam_dens # 確率密度
)

# μとλの(同時)事前分布を作図:等高線図
ggplot() + 
  geom_contour(data = prior_df, aes(x = mu, y = lambda, z = density, color = ..level.., alpha = "prior")) + # μとλの事前分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, alpha = "data"), 
             color = "red", size = 6, shape = 4) + # 真のパラメータ
  scale_alpha_manual(values = c(param = 1, prior = 1), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # (凡例表示用の黒魔術)
  guides(alpha = guide_legend(override.aes = list(shape = c(4, NA), linetype = c("blank", "solid")))) + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = paste0("list(m==", m, ", beta==", beta, ", a==", a, ", b==", b, ")")), 
       color = "density", 
       x = expression(mu), y = expression(lambda))

# μとλの(同時)事前分布を作図:塗りつぶし等高線図
ggplot() + 
  geom_contour_filled(data = prior_df, aes(x = mu, y = lambda, z = density, fill = ..level..), 
                      alpha = 0.8, size = 0) + # μとλの事前分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, color = "param"), 
             size = 6, shape = 4) + # 真のパラメータ
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = paste0("list(m==", m, ", beta==", beta, ", a==", a, ", b==", b, ")")), 
       fill = "density", 
       x = expression(mu), y = expression(lambda))


### ・事後分布(ガウス・ガンマ分布)の計算 -----

# μの事後分布のパラメータを計算:式(3.83)
beta_hat <- N + beta
m_hat    <- (sum(x_n) + beta * m) / beta_hat

# λの事後分布のパラメータを計算:式(3.88)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) + b


# μとλの(同時)事後分布を計算
posterior_df <- tidyr::tibble(
  mu = mu_lambda_mat[, 1], # 確率変数μ
  lambda = mu_lambda_mat[, 2], # 確率変数λ
  N_dens = dnorm(x = mu, mean = m_hat, sd = 1/sqrt(beta_hat*lambda)), # μの確率密度
  Gam_dens = dgamma(x = lambda, shape = a_hat, rate = b_hat), # λの確率密度
  density = N_dens * Gam_dens # 確率密度
)

# パラメータラベル用の文字列を作成
param_text <- paste0(
  "list(hat(m)==", round(m_hat, 2), ", hat(beta)==", beta_hat, 
  ", hat(a)==", a_hat, ", hat(b)==", round(b_hat, 1), ")"
)

# μとλの(同時)事後分布を作図:等高線図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = mu, y = lambda, z = density, color = ..level.., alpha = "posterior")) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, alpha = "param"), 
             color = "red", size = 6, shape = 4) + # 真のパラメータ
  scale_alpha_manual(values = c(param = 1, posterior = 1), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # (凡例表示用の黒魔術)
  guides(alpha = guide_legend(override.aes = list(shape = c(4, NA), linetype = c("blank", "solid")))) + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = param_text), 
       color = "density", 
       x = expression(mu), y = expression(lambda))

# μとλの(同時)事後分布を作図:塗りつぶし等高線図
ggplot() + 
  geom_contour_filled(data = posterior_df, aes(x = mu, y = lambda, z = density, fill = ..level..), 
                      alpha = 0.8) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, color = "param"), 
             size = 6, shape = 4) + # 真のパラメータ
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = param_text), 
       fill = "density", 
       x = expression(mu), y = expression(lambda))


### ・予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_st_hat     <- m_hat
lambda_st_hat <- beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_st_hat     <- 2 * a_hat
#mu_st_hat     <- (sum(x_n) + beta * m) / (N + beta)
#numer_lambda  <- (N + beta) * (N / 2 + a)
#denom_lambda  <- (N + 1 + beta) * ((sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b)
#lambda_st_hat <- numer_lambda / denom_lambda
#nu_st_hat     <- N + 2 * a

# 予測分布を計算:式(3.76)
predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = LaplacesDemon::dst(x = x_vec, mu = mu_st_hat, sigma = 1/sqrt(lambda_st_hat), nu = nu_st_hat) # 確率密度
)

# パラメータラベル用の文字列を作成
param_text <- paste0(
  "list(N==", N, ", hat(mu)[s]==", round(mu_st_hat, 2), 
  ", hat(lambda)[s]==", round(lambda_st_hat, 5), ", hat(nu)[s]==", nu_st_hat, ")"
)

# 予測分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  geom_line(data = predict_df, mapping = aes(x = x, y = dens, color = "predict"), 
            size = 1) + # 予測分布
  scale_color_manual(values = c(model = "red", predict = "purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Student's t Distribution", 
       subtitle = parse(text = param_text), 
       x = "x", y = "density")


# アニメーションによる学習推移の可視化 --------------------------------------------------------------

### ・モデルの設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01

# μの事前分布のパラメータを指定
m    <- 0
beta <- 1

# λの事前分布のパラメータを指定
a <- 1
b <- 1

# データ数(試行回数)を指定
N <- 100


# グラフ用のμの値を作成
mu_vec <- seq(mu_truth - 30, mu_truth + 30, length.out = 201)

# グラフ用のλの値を作成
lambda_vec <- seq(0, lambda_truth * 4, length.out = 201)

# グラフ用のμとλの点を作成
mu_lambda_mat <- tidyr::expand_grid(mu = mu_vec, lambda = lambda_vec) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# グラフ用のxの値を作成
x_vec <- seq(
  mu_truth - 1/sqrt(lambda_truth) * 4, 
  mu_truth + 1/sqrt(lambda_truth) * 4, 
  length.out = 251
)


### ・推論処理：for関数による処理 -----

# μとλの事後分布(ガウス・ガンマ分布)を計算
anime_posterior_df <- tidyr::tibble(
  mu = mu_lambda_mat[, 1], # 確率変数μ
  lambda = mu_lambda_mat[, 2], # 確率変数λ
  N_dens = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)), # μの確率密度
  Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
  density = N_dens * Gam_dens, # 確率密度
  param = paste0("N=", 0, ", m=", m, ", beta=", beta, ", a=", a, ", b=", b) |> 
    as.factor() # フレーム切替用ラベル
)

# 初期値による予測分布のパラメータを計算:式(3.95)
mu_st     <- m
lambda_st <- beta * a / (1 + beta) / b
nu_st     <- 2 * a

# 初期値による予測分布(スチューデントのt分布)を計算:式(3.76)
anime_predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
  param = paste0("N=", 0, ", mu_s=",mu_st, ", lambda_s=", lambda_st, ", nu_s=", nu_st) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_n <- rep(NA, times = N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = 1/sqrt(lambda_truth))
  
  # μの事後分布のパラメータを更新:式(3.83)
  beta_old <- beta
  m_old    <- m
  beta     <- 1 + beta
  m        <- (x_n[n] + beta_old * m) / beta
  
  # λの事後分布のパラメータを更新:式(3.88)
  a <- 0.5 + a
  b <- 0.5 * (x_n[n]^2 + beta_old * m_old^2 - beta * m^2) + b
  
  # μとλの事後分布(ガウス・ガンマ分布)を計算
  tmp_posterior_df <- tidyr::tibble(
    mu = mu_lambda_mat[, 1], # 確率変数μ
    lambda = mu_lambda_mat[, 2], # 確率変数λ
    N_dens = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)), # μの確率密度
    Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
    density = N_dens * Gam_dens, # 確率密度
    param = paste0("N=", n, ", m=", round(m, 2), ", beta=", beta, ", a=", a, ", b=", round(b, 1)) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.95)
  mu_st     <- m
  lambda_st <- beta * a / (1 + beta) / b
  nu_st     <- 2 * a
  
  # 予測分布(スチューデントのt分布)を計算:式(3.76)
  tmp_predict_df <- tibble::tibble(
    x = x_vec, # 確率変数
    dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
    param = paste0(
      "N=", n, ", mu_s=", round(mu_st, 2), ", lambda_s=", round(lambda_st, 5), ", nu_s=", nu_st
    ) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 推論結果を結合
  anime_posterior_df <- rbind(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- rbind(anime_predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### ・推論処理：tidyverseパッケージによる処理 -----

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda_truth))


# 試行ごとに事後分布と予測分布のパラメータを計算
anime_param_df <- tibble::tibble(
  # 計算用の値
  n = 0:N, 
  sum_x = cumsum(c(0, x_n)), 
  sum_x2 = cumsum(c(0, x_n)^2), 
  # μの事後分布のパラメータ:式(3.83)
  beta_hat = n + beta, 
  m_hat = (sum_x + beta*m) / (n + beta), 
  # λの事後分布のパラメータ:式(3.88)
  a_hat = 0.5*n + a, 
  b_hat = 0.5 * (sum_x2 + beta*m^2 - beta_hat*m_hat^2) + b, 
  # 予測分布のパラメータ:式(3.95)
  mu_st_hat = m_hat, 
  lambda_st_hat = beta_hat * a_hat / (1 + beta_hat) / b_hat, 
  nu_st_hat = 2 * a_hat
) |> # パラメータを計算
  dplyr::select(!c(sum_x, sum_x2)) # 不要な列を削除

# 試行ごとにμとλの(同時)事後分布を計算
anime_posterior_df <- tidyr::expand_grid(
  anime_param_df |> 
    dplyr::select(n, m = m_hat, beta = beta_hat, a = a_hat, b = b_hat), # 利用するパラメータを取得
  mu = mu_vec, # 確率変数μ
  lambda = lambda_vec # 確率変数λ
) |> # パラメータごとに確率変数(格子点)を複製
  dplyr::mutate(
    N_dens = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)), # μの確率密度
    Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
    density = N_dens * Gam_dens, # 確率密度
    param = paste0("N=", n, ", m=", round(m, 2), ", beta=", beta, ", a=", a, ", b=", round(b, 1)) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  )

# 試行ごとに予測分布(スチューデントのt分布)を計算
anime_predict_df <- tidyr::expand_grid(
  anime_param_df |> 
    dplyr::select(n, mu_st = mu_st_hat, lambda_st = lambda_st_hat, nu_st = nu_st_hat), # 利用するパラメータを取得
  x = x_vec # 確率変数
) |> # パラメータごとに確率変数を複製
  dplyr::mutate(
    dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
    param = paste0("N=", n, ", mu_s=", round(mu_st, 2), ", lambda_s=", round(lambda_st, 5), ", nu_s=", nu_st) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) # 事後分布を計算:式(3.76)


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  scaled_x = c(NA, 1/(x_n - mu_truth)^2), # 偏差の2乗の逆数に変換
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# μとλの(同時)事後分布のアニメーションを作図:等高線図
posterior_graph <- ggplot() + 
  geom_contour(data = anime_posterior_df, mapping = aes(x = mu, y = lambda, z = density, color = ..level.., alpha = "posterior")) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, alpha = "param"), 
             color = "red", size = 6, shape = 4) + # 真のパラメータ
  geom_point(data = anime_data_df, mapping = aes(x = x, y = scaled_x, alpha = "data"), 
             color = "pink", size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_alpha_manual(breaks = c("posterior", "param", "data"), 
                     values = c(1, 1, 1), 
                     labels = c("posterior", "true parameter", "observation data"), name = "") + # (凡例表示用の黒魔術)
  guides(alpha = guide_legend(override.aes = list(alpha = c(1, 1, 1), shape = c(NA, 4, 16), linetype = c("solid", "blank", "blank")))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(xlim = c(min(mu_vec), max(mu_vec)), ylim = c(0, max(lambda_vec))) + 
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = "{current_frame}", 
       color = "density", 
       x = expression(mu), y = expression(lambda)) # ラベル

# μとλの(同時)事後分布のアニメーションを作図:塗りつぶし等高線図
posterior_graph <- ggplot() + 
  geom_contour_filled(data = anime_posterior_df, mapping = aes(x = mu, y = lambda, z = density, fill = ..level..), 
                      alpha = 0.8) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, color = "param"), 
             size = 6, shape = 4) + # 真のパラメータ
  geom_point(data = anime_data_df, mapping = aes(x = x, y = scaled_x, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("param", "data"), 
                     values = c("red", "pink"), 
                     labels = c("true parameter", "observation data"), name = "") + # (凡例表示用の黒魔術)
  guides(color = guide_legend(override.aes = list(shape = c(4, 16)))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(xlim = c(min(mu_vec), max(mu_vec)), ylim = c(0, max(lambda_vec))) + 
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = "{current_frame}", 
       fill = "density", 
       x = expression(mu), y = expression(lambda)) # ラベル

# gif画像を作図
gganimate::animate(posterior_graph, nframes = N+1+10, end_pause=10, fps = 10, width = 800, height = 600)


# 真の分布を計算:式(2.64)
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_truth, sd = 1/sqrt(lambda_truth)) # 確率密度
)

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
)

# 観測データを複製して格納
anime_alldata_df <- tidyr::expand_grid(
  frame = 1:N, # フレーム番号
  n = 1:N # 試行回数
) |> # 全ての組み合わせを作成
  dplyr::filter(n < frame) |> # フレーム番号以前のデータを抽出
  dplyr::mutate(
    x = x_n[n], 
    param = unique(anime_predict_df[["param"]])[frame+1]
  ) # 対応するデータとラベルを抽出

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  geom_line(data = anime_predict_df, mapping = aes(x = x, y = dens, color = "predict"), 
            size = 1) + # 予測分布
  geom_point(data = anime_alldata_df, mapping = aes(x = x, y = 0), 
             color = "pink", alpha = 0.5, size = 3) + # 過去の観測データ
  geom_point(data = anime_data_df, mapping = aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("model", "predict", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true model", "predict", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 3), 
                                                  linetype = c("dashed", "solid", "blank"), 
                                                  shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(ylim = c(0, max(model_df[["dens"]])*2)) + # 軸の表示範囲
  labs(title = "Student's t Distribution", 
       subtitle = "{current_frame}", 
       x = "x", y = "density")

# gif画像を作図
gganimate::animate(predict_graph, nframes = N+1+10, end_pause=10, fps = 10, width = 800, height = 600)


