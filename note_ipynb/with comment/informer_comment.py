```python
# -*- coding: utf-8 -*-
"""Informer.ipynb

این فایل به‌طور خودکار توسط Colab تولید شده است.

فایل اصلی در لینک زیر قرار دارد:
    https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC

# دمو از مدل Informer

## دانلود کدها و داده‌ها
"""

# کلون کردن کدها و داده‌های مورد نیاز از گیت‌هاب
!git clone https://github.com/zhouhaoyi/Informer2020.git
!git clone https://github.com/zhouhaoyi/ETDataset.git
!ls

# بررسی و اضافه کردن مسیر کتابخانه Informer به سیستم
import sys
if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']

# نصب نیازمندی‌های پروژه (غیرفعال شده)
# !pip install -r ./Informer2020/requirements.txt

"""## آزمایش‌ها: آموزش و تست"""

# وارد کردن کتابخانه‌های مورد نیاز
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch

# تنظیمات اولیه به کمک دیکشنری
args = dotdict()

# انتخاب مدل مورد استفاده
args.model = 'informer'  # مدل: [informer, informerstack, informerlight(TBD)]

# مشخص کردن داده‌های ورودی
args.data = 'ETTh1'  # نوع داده
args.root_path = './ETDataset/ETT-small/'  # مسیر اصلی داده‌ها
args.data_path = 'ETTh1.csv'  # نام فایل داده‌ها
args.features = 'M'  # نوع پیش‌بینی: [M: چند متغیره، S: تک‌متغیره، MS: ترکیبی]
args.target = 'OT'  # ویژگی هدف در وظایف تک‌متغیره یا ترکیبی
args.freq = 'h'  # فرکانس برای کدگذاری ویژگی‌های زمانی (مثلاً ساعتی)
args.checkpoints = './informer_checkpoints'  # مسیر ذخیره مدل‌ها

# تنظیم طول توالی‌ها
args.seq_len = 96  # طول توالی ورودی برای کدگذار
args.label_len = 48  # طول برچسب‌ها برای دیکودر
args.pred_len = 24  # طول توالی پیش‌بینی
# ورودی دیکودر: [توالی برچسب‌ها + سری صفر]

# مشخص کردن ابعاد ورودی و خروجی مدل
args.enc_in = 7  # تعداد ویژگی‌های ورودی کدگذار
args.dec_in = 7  # تعداد ویژگی‌های ورودی دیکودر
args.c_out = 7  # تعداد ویژگی‌های خروجی
args.factor = 5  # ضریب فاکتور در Attention
args.d_model = 512  # ابعاد مدل
args.n_heads = 8  # تعداد سرهای Attention
args.e_layers = 2  # تعداد لایه‌های کدگذار
args.d_layers = 1  # تعداد لایه‌های دیکودر
args.d_ff = 2048  # ابعاد لایه Fully Connected
args.dropout = 0.05  # میزان Dropout
args.attn = 'prob'  # نوع Attention: [prob, full]
args.embed = 'timeF'  # روش کدگذاری زمانی: [timeF, fixed, learned]
args.activation = 'gelu'  # نوع Activation
args.distil = True  # استفاده از Distillation در کدگذار
args.output_attention = False  # آیا خروجی Attention لازم است؟
args.mix = True
args.padding = 0
args.freq = 'h'

# تنظیمات مربوط به آموزش
args.batch_size = 32
args.learning_rate = 0.0001
args.loss = 'mse'  # تابع خطا: خطای میانگین مربعات
args.lradj = 'type1'
args.use_amp = False  # استفاده از آموزش دقت مخلوط

# تنظیمات سیستم
args.num_workers = 0
args.itr = 1
args.train_epochs = 6
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# تنظیمات داده‌ها بر اساس نوع آنها
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

# چاپ تنظیمات
print('تنظیمات در آزمایش:')
print(args)

# ایجاد نمونه آزمایش
Exp = Exp_Informer

for ii in range(args.itr):
    # تنظیم رکورد آزمایش
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

    # اجرای آزمایش
    exp = Exp(args)

    # آموزش مدل
    print('>>>>>>> شروع آموزش: {} >>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    # تست مدل
    print('>>>>>>> تست مدل: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    # آزادسازی حافظه GPU
    torch.cuda.empty_cache()
```
```python
"""## پیش‌بینی"""

import os

# تنظیم مسیر ذخیره مدل
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
# مسیر ذخیره مدل قبلاً آموزش داده شده
# path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

# اگر مدل از پیش آموزش داده شده دارید، می‌توانید آرگومان‌ها و مسیر مدل را تنظیم کنید، سپس یک Experiment ایجاد کرده و از آن برای پیش‌بینی استفاده کنید
# پیش‌بینی یک دنباله است که به آخرین تاریخ داده‌ها متصل است و در داده‌ها وجود ندارد
# برای اطلاعات بیشتر در مورد پیش‌بینی می‌توانید به کد `exp/exp_informer.py` تابع `predict()` و `data/data_loader.py` کلاس `Dataset_Pred` مراجعه کنید

exp = Exp(args)

exp.predict(setting, True)

# نتیجه پیش‌بینی در مسیر ./results/{setting}/real_prediction.npy ذخیره خواهد شد
import numpy as np

prediction = np.load('./results/'+setting+'/real_prediction.npy')

prediction.shape

"""### جزئیات بیشتر درباره پیش‌بینی - تابع پیش‌بینی"""

# کد دقیق تابع پیش‌بینی در اینجا آمده است

def predict(exp, setting, load=False):
    # دریافت داده‌ها و loader برای پیش‌بینی
    pred_data, pred_loader = exp._get_data(flag='pred')

    if load:
        # بارگذاری مدل از مسیر ذخیره‌شده
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

    exp.model.eval()

    preds = []

    # پردازش داده‌ها برای پیش‌بینی
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # ورودی دیکودر
        if exp.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
        
        # پردازش در انکودر و دیکودر
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        f_dim = -1 if exp.args.features=='MS' else 0
        batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)

        # تبدیل خروجی به numpy array
        pred = outputs.detach().cpu().numpy()#.squeeze()

        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # ذخیره نتایج
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path+'real_prediction.npy', preds)

    return preds

# همچنین می‌توانید از این تابع پیش‌بینی برای دریافت نتیجه استفاده کنید
prediction = predict(exp, setting, True)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(prediction[0,:,-1])
plt.show()

"""### جزئیات بیشتر درباره پیش‌بینی - مجموعه داده‌های پیش‌بینی

شما می‌توانید `root_path` و `data_path` داده‌هایی که می‌خواهید پیش‌بینی کنید، و همچنین `seq_len`، `label_len`، `pred_len` و آرگومان‌های دیگر را مانند سایر مجموعه داده‌ها تنظیم کنید. تفاوت این است که می‌توانید یک فرکانس دقیق‌تر مانند `15min` یا `3h` برای ایجاد سری زمانی پیش‌بینی تنظیم کنید.

`Dataset_Pred` فقط یک نمونه دارد (شامل `encoder_input: [1, seq_len, dim]`، `decoder_token: [1, label_len, dim]`، `encoder_input_timestamp: [1, seq_len, date_dim]`، `decoder_input_timstamp: [1, label_len+pred_len, date_dim]`). این تابع آخرین دنباله داده داده‌شده (داده‌های seq_len) را جدا می‌کند تا دنباله آینده‌ای که دیده نشده است (داده‌های pred_len) پیش‌بینی شود.
"""

from data.data_loader import Dataset_Pred
from torch.utils.data import DataLoader

# تنظیمات مجموعه داده‌های پیش‌بینی
Data = Dataset_Pred
timeenc = 0 if args.embed!='timeF' else 1
flag = 'pred'; shuffle_flag = False; drop_last = False; batch_size = 1

freq = args.detail_freq

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
)
# ایجاد DataLoader
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

len(data_set), len(data_loader)
```
```python
"""## مصورسازی"""

# وقتی که آموزش مدل (exp.train(setting)) و تست مدل (exp.test(setting)) تمام شود، یک مدل آموزش‌دیده و نتایج آزمایش تست دریافت خواهیم کرد.
# نتایج آزمایش تست در مسیر ./results/{setting}/pred.npy (پیش‌بینی مجموعه تست) و ./results/{setting}/true.npy (مقادیر واقعی مجموعه تست) ذخیره می‌شوند.

preds = np.load('./results/'+setting+'/pred.npy')  # بارگذاری مقادیر پیش‌بینی شده
trues = np.load('./results/'+setting+'/true.npy')  # بارگذاری مقادیر واقعی

# شکل داده‌ها: [تعداد نمونه‌ها، طول پیش‌بینی، تعداد ابعاد]
preds.shape, trues.shape

import matplotlib.pyplot as plt
import seaborn as sns

# رسم پیش‌بینی OT
plt.figure()
plt.plot(trues[0,:,-1], label='GroundTruth')  # مقادیر واقعی
plt.plot(preds[0,:,-1], label='Prediction')  # مقادیر پیش‌بینی شده
plt.legend()
plt.show()

# رسم پیش‌بینی HUFL
plt.figure()
plt.plot(trues[0,:,0], label='GroundTruth')  # مقادیر واقعی
plt.plot(preds[0,:,0], label='Prediction')  # مقادیر پیش‌بینی شده
plt.legend()
plt.show()

# بارگذاری داده‌ها برای آزمایش
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader

Data = Dataset_ETT_hour  # انتخاب نوع مجموعه داده
timeenc = 0 if args.embed!='timeF' else 1  # تنظیم کدگذاری زمانی
flag = 'test'  # نوع داده (تست)
shuffle_flag = False  # عدم شافل کردن داده‌ها
drop_last = True  # حذف آخرین دسته ناقص
batch_size = 1  # تعداد نمونه‌ها در هر دسته

# تنظیم مجموعه داده تست
data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    freq=args.freq
)

# ایجاد DataLoader برای مدیریت داده‌ها
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last
)

import os

args.output_attention = True  # فعال کردن توجه (Attention)

exp = Exp(args)  # ایجاد آزمایش

model = exp.model  # دریافت مدل

# بارگذاری مدل آموزش‌دیده از مسیر
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
model.load_state_dict(torch.load(path))

# مصورسازی توجه (Attention)
idx = 0  # انتخاب نمونه
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
    if i != idx:  # ادامه تا رسیدن به نمونه انتخابی
        continue
    batch_x = batch_x.float().to(exp.device)  # انتقال ورودی به دستگاه (مثلاً GPU)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(exp.device)  # انتقال نشانگر زمان ورودی به دستگاه
    batch_y_mark = batch_y_mark.float().to(exp.device)  # انتقال نشانگر زمان خروجی به دستگاه

    # آماده‌سازی ورودی دیکودر
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

    # اجرای مدل برای دریافت خروجی و توجه
    outputs, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# بررسی شکل توجه‌ها (Attention maps)
attn[0].shape, attn[1].shape  # , attn[2].shape

# مصورسازی توجه در لایه اول
layer = 0  # لایه انتخاب شده
distil = 'Distil' if args.distil else 'NoDistil'  # وضعیت کاهش ابعاد
for h in range(0, 8):  # رسم نقشه توجه برای هر سر (head)
    plt.figure(figsize=[10, 8])
    plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
    A = attn[layer][0, h].detach().cpu().numpy()
    ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
    plt.show()

# مصورسازی توجه در لایه دوم
layer = 1  # تغییر لایه به دوم
distil = 'Distil' if args.distil else 'NoDistil'
for h in range(0, 8):  # رسم نقشه توجه برای هر سر (head)
    plt.figure(figsize=[10, 8])
    plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
    A = attn[layer][0, h].detach().cpu().numpy()
    ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
    plt.show()
```
```python
"""## داده‌های سفارشی

داده‌های سفارشی (فایل xxx.csv) باید حداقل شامل ۲ ویژگی باشند: `date` (در فرمت: `YYYY-MM-DD hh:mm:ss`) و ویژگی هدف (`target feature`).

"""

# بارگذاری کلاس Dataset_Custom برای مدیریت داده‌ها
from data.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
import pandas as pd
import os

# داده‌های سفارشی: فایل xxx.csv
# ویژگی‌های داده: ['date', ...(سایر ویژگی‌ها), ویژگی هدف]

# به عنوان مثال از مجموعه داده ETTh2 استفاده می‌کنیم
args.root_path = './ETDataset/ETT-small/'  # مسیر اصلی مجموعه داده
args.data_path = 'ETTh2.csv'  # نام فایل مجموعه داده

# بارگذاری داده‌های CSV
df = pd.read_csv(os.path.join(args.root_path, args.data_path))

# نمایش پنج سطر اول داده‌ها برای بررسی
df.head()

'''
ویژگی 'HULL' را به‌عنوان هدف (target) تنظیم می‌کنیم به جای 'OT'

فرکانس‌های پشتیبانی شده شامل موارد زیر هستند:
        Y   - سالانه
            نام مستعار: A
        M   - ماهانه
        W   - هفتگی
        D   - روزانه
        B   - روزهای کاری
        H   - ساعتی
        T   - دقیقه‌ای
            نام مستعار: min
        S   - ثانیه‌ای
'''

# تنظیم ویژگی هدف به 'HULL'
args.target = 'HULL'

# تنظیم فرکانس به 'h' (ساعتی)
args.freq = 'h'

# ایجاد Dataset_Custom برای داده‌های سفارشی
Data = Dataset_Custom
timeenc = 0 if args.embed != 'timeF' else 1  # تنظیم کدگذاری زمانی
flag = 'test'  # نوع داده (تست)
shuffle_flag = False  # عدم شافل کردن داده‌ها
drop_last = True  # حذف آخرین دسته ناقص
batch_size = 1  # اندازه دسته داده‌ها

# پیکربندی مجموعه داده
data_set = Data(
    root_path=args.root_path,  # مسیر اصلی داده‌ها
    data_path=args.data_path,  # مسیر فایل داده‌ها
    flag=flag,  # نوع داده
    size=[args.seq_len, args.label_len, args.pred_len],  # تنظیمات طول دنباله‌ها
    features=args.features,  # ویژگی‌های داده
    timeenc=timeenc,  # استفاده یا عدم استفاده از کدگذاری زمانی
    target=args.target,  # ویژگی هدف (در اینجا 'HULL')
    freq=args.freq  # فرکانس (در اینجا 'h': ساعتی، 't': دقیقه‌ای)
)

# ایجاد DataLoader برای مدیریت مجموعه داده
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,  # تعداد نمونه‌ها در هر دسته
    shuffle=shuffle_flag,  # عدم شافل کردن داده‌ها
    num_workers=args.num_workers,  # تعداد رشته‌های موازی
    drop_last=drop_last  # حذف آخرین دسته ناقص
)

# دریافت اولین نمونه از مجموعه داده (ورودی‌ها و نشانگرهای زمانی)
batch_x, batch_y, batch_x_mark, batch_y_mark = data_set[0]
```