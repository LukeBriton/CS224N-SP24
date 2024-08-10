@echo off
setlocal

set DATASETS_DIR=utils\datasets
mkdir %DATASETS_DIR%

cd %DATASETS_DIR%

REM Get Stanford Sentiment Treebank
where wget >nul 2>&1
if %ERRORLEVEL%==0 (
    wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
) else (
    curl -L http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -o stanfordSentimentTreebank.zip
)

unzip stanfordSentimentTreebank.zip
del stanfordSentimentTreebank.zip

endlocal
