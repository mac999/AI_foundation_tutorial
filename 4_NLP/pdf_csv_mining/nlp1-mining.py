from fpdf import FPDF
import lorem 
import pandas as pd

# 표 출력을 위해, 데이터 프레임 생성
df = pd.DataFrame(
          {'feature 1' : ['cat 1', 'cat 2', 'cat 3', 'cat 4'],
           'feature 2' : [400, 300, 200, 100]
          })

# 차트 그림 생성
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1,1, figsize = (6, 4))
sns.barplot(data =  df, x = 'feature 1', y = 'feature 2')
plt.title("Chart")
plt.savefig('./example_chart.png', 
           transparent=False,  
           facecolor='white', 
           bbox_inches="tight")

# PDF 클래스 생성
ch = 8
class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'Header', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

# 셀을 생성하고, 각 셀에 보고서 내용 설정
pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 24)
pdf.cell(w=0, h=20, txt="Title", ln=1)
pdf.set_font('Arial', '', 16)
pdf.cell(w=30, h=ch, txt="Date: ", ln=0)
pdf.cell(w=30, h=ch, txt="01/01/2022", ln=1)
pdf.cell(w=30, h=ch, txt="Author: ", ln=0)
pdf.cell(w=30, h=ch, txt="Max Mustermann", ln=1)
pdf.ln(ch)
pdf.multi_cell(w=0, h=5, txt=lorem.paragraph())
pdf.image('./example_chart.png', x = 10, y = None, w = 100, h = 0, type = 'PNG', link = '') # 차트 이미지 입력
pdf.ln(ch)
pdf.multi_cell(w=0, h=5, txt=lorem.paragraph())
pdf.ln(ch)

# 표 헤더 및 내용 설정
pdf.set_font('Arial', 'B', 16)
pdf.cell(w=40, h=ch, txt='Feature 1', border=1, ln=0, align='C')
pdf.cell(w=40, h=ch, txt='Feature 2', border=1, ln=1, align='C')
# Table contents
pdf.set_font('Arial', '', 16)
for i in range(len(df)):
    pdf.cell(w=40, h=ch, txt=df['feature 1'].iloc[i], border=1, ln=0, align='C')
    pdf.cell(w=40, h=ch, txt=df['feature 2'].iloc[i].astype(str), border=1, ln=1, align='C')

# PDF 저장
pdf.output("example.pdf")