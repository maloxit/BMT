from bot.bot import Bot
import argparse


argperser = argparse.ArgumentParser()
argperser.add_argument('--checkpoint', type=str, default='weights/BMT_NND_WM_FSPL100.tar', help='Model checkpoint path')
argperser.add_argument('--token', type=str, help='Telegram API Token', required=True)

opts = argperser.parse_args()

bot = Bot(opts.token, opts.checkpoint)
bot.run()