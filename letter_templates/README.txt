Letter templates for the word-search bot
========================================

YES — screenshots of each letter help a lot more than the "more training"
error pairs. EasyOCR is generic; your game font is specific.

HOW TO SAVE
-----------
1. Crop ONE clean letter cell (or tight crop around one letter).
2. Save into this folder with the letter as the filename:

     A.png   B.png   C.png  ...  Z.png

3. Optional extras (multiple samples of the same letter are great):

     A.png   A2.png   A_clean.png   a.png
     (letter + optional digits / _suffix only)

4. Prefer:
   - same game font / size / color as live play
   - full cell or tight crop, not tiny sliver
   - light background + dark letter (as in the game)

5. After adding files, restart the bot (templates load on start).

DO NOT put full board screenshots here as templates.
Capture.PNG / full UI dumps are ignored (they used to load as letter "C").

HIDDEN WORDS (game rule from the tutorial popup)
-----------------------------------------------
"Some levels have HIDDEN WORDS that appear as solid circles in the word bank!"

Teaching samples in fag/:
  Capture.PNG  — pure row of solid black ●●●●●● (entire bank is hidden)
  2.PNG        — mid-gray ● mixed with visible text (e.g. letter + ●●●●)
  1.PNG        — CLEANING SUPPLIES: all words visible + green leaf gems

Those solid ● slots are NOT text OCR. The bot:
  - counts solid black AND mid-gray circular placeholders (scale-adaptive)
  - sums multi-row banks (not just the longest single row)
  - ignores green gems/leaves, gold coins, and any 0/N counter
  - solves the visible words (SUMMER, FREEZER, BAG, …)
  - then searches the grid for N extra dictionary/theme words
  - yellow gold gems ON the letter grid are whitened before cell OCR

WHAT HAPPENS
------------
The bot matches each cell against these templates FIRST.
EasyOCR is only a backup when templates are missing or a weak match.

You do NOT need all 26 on day one — add the ones it misreads most
(B/R, I/L, E/F, C/G/S, etc.) then fill the rest.
