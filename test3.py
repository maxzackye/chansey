import chansey
import numpy as np
import pandas as pd
import warnings
import time
from datetime import date, timedelta, datetime
from tmlpatch.database import TMLSQLClient
warnings.filterwarnings("ignore")


def get_data():
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')

    sql1 = f"""
            select fund_name
            ,light_credit_available_limit_td as `总授信额度`
            ,light_loan_apply_amount_1d as `总发起金额`
            ,light_loan_success_light_loan_principal_1d as `总动支金额`
            ,date(to_date(ds,'yyyyMMdd')) as `日期`
            from ads_app_metricstore.mts_explore_3_1_0_ind_2964950
            where ds >= '20240210'
            """
    
    sql2 = """
            select a.day as `日期`
            ,if(a.light_diversion_scene='贷前','贷前授信拒绝',a.light_diversion_scene) as `场景`
            ,b.light_diversion_usr_cnt as `总人数`
            ,b.light_diversion_count as `总进入路由次数`
            ,b.light_enter_filter_count as `总进入前筛次数`
            ,b.light_enter_filter_pass_light_enter_filter_count as `总前筛通过次数`
            ,a.fund_name as `资方`
            ,c.light_access_count as `总进入准入次数`
            ,c.light_access_pass_light_access_count as `总准入通过次数`
            ,a.light_credit_audit_count as `总送审次数`
            ,a.light_credit_audit_success_light_credit_audit_count as `总授信通过次数`
            ,a.light_init_credit_limit as `新增授信额度`
            from
            (select date(to_date(ds,'yyyyMMdd')) as day
            ,fund_name
            ,case when light_diversion_scene_record not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene_record end as light_diversion_scene
            ,sum(light_credit_audit_count_1d) as light_credit_audit_count
            ,sum(light_credit_audit_success_light_credit_audit_count_1d) as light_credit_audit_success_light_credit_audit_count
            ,sum(light_init_credit_limit_1d) as light_init_credit_limit
            from ads_app_metricstore.mts_explore_3_2_0_ind_0538590
            where ds >= '20240210'
            group by date(to_date(ds,'yyyyMMdd'))
            ,fund_name
            ,case when light_diversion_scene_record  not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene_record end
            )a
            left join
            (select date(to_date(ds,'yyyyMMdd')) as day
            ,case when light_diversion_scene not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene end as light_diversion_scene
            ,sum(light_diversion_usr_cnt_1d) as light_diversion_usr_cnt
            ,sum(light_diversion_count_1d) as light_diversion_count
            ,sum(light_enter_filter_count_1d) as light_enter_filter_count
            ,sum(light_enter_filter_pass_light_enter_filter_count_1d) as light_enter_filter_pass_light_enter_filter_count
            from ads_app_metricstore.mts_explore_4_1_0_ind_0056940
            where ds >= '20240210'
            group by date(to_date(ds,'yyyyMMdd'))
            ,case when light_diversion_scene not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene end
            )b
            on a.day = b.day and a.light_diversion_scene = b.light_diversion_scene
            left join 
            (select date(to_date(ds,'yyyyMMdd')) as day
            ,fund_name
            ,case when light_diversion_scene not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene end as light_diversion_scene
            ,sum(light_access_count_1d) as light_access_count --总进入准入次数
            ,sum(light_access_pass_light_access_count_1d) as light_access_pass_light_access_count --总准入通过次数
            from ads_app_metricstore.mts_explore_5_2_0_ind_6465050
            where ds >= '20240210'
            group by date(to_date(ds,'yyyyMMdd'))
            ,fund_name
            ,case when light_diversion_scene not in ('贷前','轻资产提额','轻资产自动送审','轻资产预借款') then '其他' else light_diversion_scene end
            )c
            on a.day = c.day and a.light_diversion_scene = c.light_diversion_scene and a.fund_name = c.fund_name
            """
    
    sql3 = """
            select a.day as `日期`
            ,a.fund_name as `资方`
            ,'存量授信额度' as `新老户`
            ,a.light_credit_available_limit_td-coalesce(b.light_init_credit_limit,0) as `总授信额度`
            from
            (select fund_name
            ,light_credit_available_limit_td
            ,date(to_date(ds,'yyyyMMdd')) as day
            from ads_app_metricstore.mts_explore_3_1_0_ind_2964950
            where ds >= '20240210')a
            left join 
            (select date(to_date(ds,'yyyyMMdd')) as day
            ,fund_name
            ,sum(light_init_credit_limit_1d) as light_init_credit_limit
            from ads_app_metricstore.mts_explore_3_2_0_ind_0538590
            where ds >= '20240210'
            group by date(to_date(ds,'yyyyMMdd'))
            ,fund_name
            )b
            on a.day = b.day and a.fund_name = b.fund_name 

            union all

            select date(to_date(ds,'yyyyMMdd')) as `日期`
            ,fund_name  as `资方`
            ,'新增授信额度' as `新老户`
            ,sum(light_init_credit_limit_1d) as `总授信额度`
            from ads_app_metricstore.mts_explore_3_2_0_ind_0538590
            where ds >= '20240210'
            group by date(to_date(ds,'yyyyMMdd'))
            ,fund_name
            """
    client = TMLSQLClient()
    df1 = client.sql(sql1).to_pandas()
    df2 = client.sql(sql2).to_pandas()
    df3 = client.sql(sql3).to_pandas()
    return df1,df2,df3

def save_detail_to_dp(output_df):
    '''
    保存详情到dp表
    '''
    save_table = 'pdm_ailab.pdm_ailab_chansey_light_business_day_output_di'
    tmp_save_table = save_table + str(int(time.time() * 1e6))
    client = TMLSQLClient()
    client.sql(f'drop table if exists {tmp_save_table}')
    client.to_table(output_df, tmp_save_table)

    col_schema = []
    for col in output_df.columns:
        if col in 'ds':
            continue
        else:
            col_schema.append(f'{col} STRING')

    client.sql(f'''
    CREATE TABLE IF NOT EXISTS {save_table}(
    {','.join(col_schema)}
    )    PARTITIONED BY (ds STRING COMMENT '日分区字段,格式yyyymmdd') STORED AS ALIORC TBLPROPERTIES ('comment' = '轻资产动支金额交易路由分诊器');  
    ''')

    client.sql(f'''
    insert into table {save_table} PARTITION (ds)
    select * from {tmp_save_table}
    ''')
    try:
        client.sql(f'drop table if exists {tmp_save_table}')
    except Exception as e:
        import traceback
        traceback.print_exc()



def calculate():
    # 准备数据
    # df1,df2,df3 = get_data()
    df1 = pd.read_csv(r'C:\Users\Administrator\git_project\交易漏斗.csv')
    df2 = pd.read_csv(r'C:\Users\Administrator\git_project\路由漏斗.csv')
    df3 = pd.read_csv(r'C:\Users\Administrator\git_project\授信额度拆分.csv')
    dict1 = chansey.MultiDataset({'交易漏斗':(df1,'日期'),'路由漏斗':(df2,'日期'),'授信额度拆分':(df3,'日期')})

    fund_list = df1['资方'].drop_duplicates().to_list()
    scene_list = df2['场景'].drop_duplicates().to_list()
    day_list = df1['日期'].drop_duplicates().to_list()

    # 构造拆解树
    ## 定义各资方下漏斗分析节点的函数
    def flow1(fund):
        flow1 = chansey.Funnel(name='总动支金额',
                            children=[chansey.NumberMetric(name='总授信额度')],
                                )
        flow1.children.extend(
            chansey.generate_metrics_by_funnel(
                number_names=['总授信额度', '总发起金额', '总动支金额'],
                ratio_names=['金额发起率', '金额通过率']
            )[1]
        )
        return flow1

    ## 构造第一层维度拆解（by资方）和第二层漏斗拆解
    business_formula = chansey.NumberMetric(name='总动支金额',dimension='资方'
                            ,dataset = dict1,dataset_key='交易漏斗'
                            ,children={str(key):flow1(key) for key in fund_list}
                            )

    ## 构造第三层（by场景）维度拆解
    for fund in fund_list:
        business_formula.children[str(fund)].children[0] = chansey.NumberMetric(name='总授信额度'
                                                                        ,dimension='新老户'
                                                                        ,dataset_key = '授信额度拆分'  # 第三层时更换了数据集key
                                                                        ,children={'新增授信额度':chansey.NumberMetric(name = '新增授信额度'
                                                                                                                    ,dimension = '场景'
                                                                                                                    ,dataset_key = '路由漏斗'
                                                                                                                    ,children = {str(key):chansey.Funnel(name = '新增授信额度',children=[chansey.NumberMetric(name='总人数')]) for key in scene_list}) ,
                                                                                   '存量授信额度':chansey.NumberMetric(name = '存量授信额度')}
                                                                            )
        for i in ['贷前授信拒绝','轻资产提额','轻资产自动送审']:
            business_formula.children[fund].children[0].children['新增授信额度'].children[i].children.extend(
            chansey.generate_metrics_by_funnel(
                number_names=['总人数', '总进入路由次数', '总进入前筛次数', '总前筛通过次数', '总进入准入次数', '总准入通过次数', '总送审次数', '总授信通过次数', '新增授信额度'],
                ratio_names=['人均路由次数','进入前筛率','前筛通过率','进入准入率','准入通过率','送审率','授信率','平均授信额度']
            )[1]
            )
        for i in ['轻资产预借款','其他']:
            business_formula.children[fund].children[0].children['新增授信额度'].children[i].children.extend(
            chansey.generate_metrics_by_funnel(
                number_names=['总人数', '总进入路由次数','总送审次数', '总授信通过次数', '新增授信额度'],
                ratio_names=['平均路由次数','送审率','授信率','平均授信额度']
            )[1]
            )

    # 获取下钻结果
    day_list = sorted(day_list)
    obs_period = [month_list[-1]]
    output_df = pd.DataFrame()
    for month in month_list[:-1]:
        cmp_period = [month]
        output = business_formula.drilldown(obs_period, cmp_period) 
        output['obs_period'] = obs_period[0]
        output['cmp_period'] = cmp_period[0]
        output.reset_index(inplace=True)
        output.rename(columns={'index':'grp'},inplace=True)
        output_df = pd.concat([output_df, output], axis=0)

        
    # 获取上个月月末的日期
    now = pd.Timestamp.now()
    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
    this_month_start = now - pd.offsets.MonthBegin(1) #+ pd.offsets.MonthEnd(0)
    last_month_end = (this_month_start - timedelta(days=1)).strftime('%Y%m%d')
    output_df['create_time'] = formatted_now
    output_df['ds'] = last_month_end
    
    # save_detail_to_dp(output_df)

    return output_df


if __name__ == '__main__':
    calculate()
