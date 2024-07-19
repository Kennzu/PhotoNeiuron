import psycopg2
import asyncio
import logging
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from token_1 import TOKEN

logging.basicConfig(level=logging.INFO)
bot = Bot(token = TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

name_tb = []

class States(StatesGroup):
    acc_can = State()
    name_table = State()

@dp.message_handler(commands='start')
async def start_message(message: types.Message):
    mark_do = types.InlineKeyboardMarkup(row_width=2)
    btn_rules = types.InlineKeyboardButton("Правила пользования", callback_data="Rules")
    btn_instr = types.InlineKeyboardButton("Инструкция", callback_data="Instruction")
    mark_do.add(btn_rules, btn_instr)
    await bot.send_message(message.chat.id, '''Приветствую вас в программном продукте для загрузки и просмотра CleanCheck!
Перед началом работы советую ознакомиться с: 
1 - Правилами пользования программным продуктом
2 - Инструкцией по взаимодействию с ботом''', reply_markup=mark_do)

@dp.callback_query_handler(text='back')
async def start_message(message: types.Message):
    mark_do = types.InlineKeyboardMarkup(row_width=2)
    btn_rules = types.InlineKeyboardButton("Правила пользования", callback_data="Rules")
    btn_instr = types.InlineKeyboardButton("Инструкция", callback_data="Instruction")
    mark_do.add(btn_rules, btn_instr)
    await bot.send_message(message.from_user.id, '''Я настоятельно рекомендую вам ознакомиться с этими пунктами. Если вы их игнорируете, то вы можете потратить больше времени на изучение и отказ в предоставлении услуг программного продукта. 
1 - Правила пользования программным продуктом
2 - Инструкция по взаимодействию с ботом''', reply_markup=mark_do)

@dp.callback_query_handler(text='Rules')
async def rd_rules(message: types.Message):
    mark_YesNo = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True, one_time_keyboard=True)
    btn_accept = types.InlineKeyboardButton("Я принимаю пользовательское соглашение")
    btn_cancel = types.InlineKeyboardButton("Я не принимаю пользовательское соглашение")
    mark_YesNo.add(btn_accept, btn_cancel)
    await bot.send_document(message.from_user.id, open(r'C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/Пользовательское_соглашение.pdf', 'rb'), reply_markup=mark_YesNo)
    await States.acc_can.set()

@dp.message_handler(state=States.acc_can)
async def accept_next(message: types.Message, state: FSMContext):
    async with state.proxy() as change:
        if message.text == "Я принимаю пользовательское соглашение":
            mark_table = types.InlineKeyboardMarkup(row_width=1)
            btn_create_table = types.InlineKeyboardButton("Создать таблицу", callback_data="TABLE")
            mark_table.add(btn_create_table)
            await bot.send_message(message.chat.id, '''Отлично!
Давай приступим к началу работы:
1 - Выбери пункт "Создать таблицу" для записи ответов и подтверждения УК''', reply_markup=mark_table)
            await state.finish()
        elif message.text == "Я не принимаю пользовательское соглашение":
            mark_back = types.InlineKeyboardMarkup(row_width=1)
            btn_back = types.InlineKeyboardButton("Вернуться назад", callback_data="back")
            mark_back.add(btn_back)
            await bot.send_message(message.from_user.id, '''Для использования данного программного продукта требуется согласиться с пользовательским соглашением. Пожалуйста, ознакомьтесь с условиями пользовательского соглашения и нажмите кнопку "Принимаю пользовательское соглашение".''', reply_markup=mark_back)
            await state.finish()

@dp.callback_query_handler(text='TABLE')
async def crt_table(message: types.Message):
    await bot.send_message(message.from_user.id, "Введите будущее название торговой точки: ")
    await States.name_table.set()

@dp.message_handler(state=States.name_table)
async def edit_name(message: types.Message, state: FSMContext):
    name_tb.append(message.text)
    a = str(name_tb[0])
    db =  psycopg2.connect(dbname='COFIX_CC', user='postgres', # ноутбук
                    password='KokoRari-23', host='localhost', port='5432')
    
    cur = db.cursor()
    try:
        cur.execute(f'''CREATE TABLE "{a}" (
                        id SERIAL NOT NULL PRIMARY KEY,
                        Вход TEXT,
                        Стол TEXT,
                        Стулья TEXT,
                        Угол_Безопасности TEXT,
                        Полы TEXT,
                        Стены TEXT,
                        Касса TEXT,
                        Кофемашина TEXT,
                        Форсунка TEXT,
                        Кофемолка TEXT,
                        Ледогенератор TEXT,
                        Витрина TEXT,
                        Полки_над_баром TEXT,
                        Кондимент TEXT,
                        Верхняя_полка_витрины TEXT,
                        Нижняя_полка_витрины TEXT,
                        Стекла_Витрины TEXT,
                        Микроволновка TEXT,
                        time date NOT NULL
            )''')
        db.commit()
        db.close()
    except psycopg2.errors.DuplicateTable:
        
        await bot.send_message(message.from_user.id, "Таблица создана!!!")



async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())