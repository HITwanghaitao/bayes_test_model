# 导入库
from docx import Document
import numpy as np
import scipy.io as io

# 创建文档对象
ua = 0.2
ub = 0.6
theta_t = 0.403
params = ['Lgrad','vc','h','m','rou']
data = np.random.randn(5,5)
def produce_docx(data,param):
    document = Document()

    # 创建5行7列表格
    table = document.add_table(rows=7, cols=5,style='Table Grid')


    table.cell(0, 0).text = '参数'
    table.cell(0, 1).text = 'theat_t'
    table.cell(0, 3).text = '先验分布'

    table.cell(1, 0).text = param
    table.cell(1, 1).text = str.format("{:.3f}",data[0,0])
    table.cell(1, 3).text = 'U['+ str.format("{:.3f}",data[0,2]) + ','+ str.format("{:.3f}",data[0,3]) + ']'


    table.cell(2, 0).text = '推断次数'
    table.cell(2, 1).text = 'map'
    table.cell(2, 2).text = 'map与theat_t的误差%'
    table.cell(2, 3).text = '95%置信区间'
    table.cell(2, 4).text = '后验分布与先验分布的区间宽度百分比%'
    for j in range(3, 7):
        table.cell(j, 0).text = str(j-2)
        table.cell(j, 1).text = str.format("{:.3f}", data[j - 2, 0])
        table.cell(j, 2).text = str.format("{:.2f}", data[j - 2, 1])
        table.cell(j, 3).text = '['+ str.format("{:.3f}", data[j-2, 2]) + ','+ str.format("{:.3f}", data[j-2, 3]) + ']'
        table.cell(j, 4).text = str.format("{:.2f}", data[j - 2, 4])
        # 保存文档
    document.save(param+'table.docx')
    return None
for  param in params:
    data = io.loadmat(param + '_data_c.mat')[param+'_c']
    produce_docx(data,param)