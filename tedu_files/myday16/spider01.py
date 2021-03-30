from urllib import request
from fake_useragent import UserAgent
import re



class SteelSpider(object):
    # 定义常用变量
    def __init__(self):
        pass

    # 发请求,url得从run()中传进来
    def get_html(self):
        url = 'http://ztweixin.steelcn.cn/Price?cityid=1080200&steeltypeid=&from=1'

        # 包装请求头
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1'}
        req = request.Request(url=url, headers=headers)
        # 发请求
        resp = request.urlopen(req)
        # 读取数据
        html = resp.read().decode()
        # print(html)

        #直接调用解析函数
        self.parse_html(html)

    # 解析函数,用正则解析提取到的页面数据
    def parse_html(self, html):
        # 创建编译对象
        re_bds = '<div class="con">.*?<li>(.*?)</li>.*?<li>(.*?)</li>.*?<li class="">(.*?)</li>.*?<li>(.*?)</li>.*?<li>(.*?)</li>'
        pattern = re.compile(re_bds, re.S)
        # 用编译对象进行正则匹配
        r_list = pattern.findall(html)
        # print(r_list)

        #保存到本地文件
        with open('steel.txt','a') as f:
            for r in r_list:
                f.write(r[0] + '\t' +
                        r[1].strip() + '\t' +
                        r[2].strip()  + '\t' +
                        r[3].strip()  + '\t' +
                        r[4].strip()  + '\n'
                        )

        print('成功')








        # self.save_html(r_list)
#
#     # 处理每一页的电影信息，最终格式为[(),(),(),...]
#     def save_html(self, r_list):
#
#         item = {}
#         for steel in r_list:
#             item['name'] = steel[0].strip()
#             item['material'] = steel[1].strip()
#             item['price'] = steel[2].strip()
#             item['standard'] = steel[3].strip()
#             item['manuf'] = steel[4].strip()
#             print(item)
#             # # 将每个电影的信息做成一个元组
#             # s = (film[0], film[1].strip(), film[2].strip()[5:15])
#             # # 放入all_list，最终格式为：[(),(),(),...]
#             # self.all_list.append(s)
#             #
#             # # 计数
#             # self.i += 1
#
#     # 入口函数
#     def run(self):
#         # 拼接url
#         url = self.url
#         # 用实例调用发送请求的函数，并传入url，接收每一页的html数据
#         html = self.get_html(url)
#         # 数据处理，用实例对象调用解析函数解析爬到的数据，并传入原数据，得到每页的最终数据
#         r_list = self.parse_html(html)
#         # r_list 数据格式:
#         # [('名称'，' xx主演xx ','上映时间：1993-01-01'),('名称'，'xx主演xx ','上映时间：1993-01-01'),...]
#
#         # 将每页数据全都放到 all_list 中
#         self.save_html(r_list)
#
#         # # 设置读取的间隔时间
#         # time.sleep(random.uniform(0, 1))
#
#         print('数据数量：', self.i)
#         # 最终写入格式：音乐之声 主演：朱莉·安德鲁斯,克里斯托弗·普卢默,埃琳诺·帕克 1965-03-02  。。。
#
#
if __name__ == '__main__':
    spider = SteelSpider()
    spider.get_html()








