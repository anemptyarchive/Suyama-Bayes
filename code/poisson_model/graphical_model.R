
# ポアソンモデル ---------------------------------------------------------------

# chapter 3.2.3
# グラフィカルモデル表現


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(DiagrammeR)
library(DiagrammeRsvg)


# グラフィカルモデルの作図 -----------------------------------------------------

# ポアソンモデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'Poisson model'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    a [label = 'a']
    b [label = 'b']
    
    lambda [label = '&lambda;']
    
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      x [label = 'x@_{n}', style = filled, filledcolor = gray]
    }
    
    {a, b} -> lambda -> x;
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 500) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/poisson_model.png", dpi = 100) # pngファイルに変換


