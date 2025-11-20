"""测试"""
import pandas as pd
import graphviz
import chansey


def test():
    # 拆解动支金额
    business_formula = chansey.Funnel('总动支金额')
    business_formula.children = [chansey.NumberMetric(name='总授信额度', dimension='procedure')]
    business_formula.children.extend(
        chansey.generate_metrics_by_funnel(
            number_names=['总授信额度', '总发起金额', '总动支金额'],
            ratio_names=['金额发起率', '金额通过率']
        )[1]
    )
    print(business_formula, end='\n\n')

    # 拆解动支金额
    business_formula.children[0].children = {
        '主流程': chansey.Funnel('总授信额度'),
        '强制导轻流程': chansey.Funnel('总授信额度')
    }
    print(business_formula, end='\n\n')

    # 拆解路由
    temp_metrics = chansey.generate_metrics_by_funnel(
        number_names=['总人数', '总前筛通过人数', '总准入通过人数', '总授信人数', '总授信额度'],
        ratio_names=['前筛通过率', '准入通过率', '授信通过率', '人均授信额度']
    )
    business_formula.children[0].children['主流程'].children = temp_metrics[0][:1] + temp_metrics[1]
                                     
    temp_metrics = chansey.generate_metrics_by_funnel(
        number_names=['总人数', '总送审人数', '总授信人数', '总授信额度'],
        ratio_names=['送审率', '授信通过率', '人均授信额度']
    )
    business_formula.children[0].children['强制导轻流程'].children = temp_metrics[0][:1] + temp_metrics[1]
    print(business_formula, end='\n\n')

    print(business_formula.to_dot(), end='\n\n')
    src = graphviz.Source(business_formula.to_dot())
    src.render(view=True, format='png')
    # 下钻



if __name__ == '__main__':
    test()
