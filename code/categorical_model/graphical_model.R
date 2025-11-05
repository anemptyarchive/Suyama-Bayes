
# カテゴリカルモデル -----------------------------------------------------------

# chapter 3.2.2
# グラフィカルモデル表現


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(DiagrammeR)
library(DiagrammeRsvg)


# グラフィカルモデルの作図 -----------------------------------------------------

# カテゴリカルモデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'Categorical model'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    alpha [label = <<B>&alpha;</B>>]
    
    phi [label = <<B>&phi;</B>>]
    
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      s [label = <<B>s</B>@_{n}>, style = filled, filledcolor = gray]
    }
    
    alpha -> phi -> s;
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 500) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/categorical_model.png", dpi = 100) # pngファイルに変換


