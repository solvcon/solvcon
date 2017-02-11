" editing behavior
set nocompatible
set modeline
set modelines=10
set noai bs=2 sw=2

" format
set ff=unix
set fileencodings=utf-8,big5hkscs,sjis,gbk,gb2312,euc-jp,euc-kr,utf-bom,iso8859-1

" appearance
set background=dark
set ruler " show the cursor position all the time
syntax on
filetype plugin on

" status line
set laststatus=2
set statusline=%f\ %h%w%m%r\ %#warningmsg#%*%=%(%{getcwd()}\ %l,%c%V\ %=\ %P%)

" search
set hlsearch
set showmatch
set wildmode=longest,list
set matchpairs+=<:>

" history
set viminfo='20,\"50 " read/write a .viminfo file, don't store more
                     " than 50 lines of registers
set history=50 " keep 50 lines of command line history

" backup
set nobackup
set nowritebackup
set noswapfile

" session
set ssop=blank,buffers,curdir,folds,help,options,resize,tabpages,winsize

" autocmd and syntax.
au FileType make set ai nu sw=8 ts=8
au FileType gitcommit set noai nu tw=999
au FileType python set ai et nu sw=4 ts=4 tw=79 omnifunc=pythoncomplete#Complete
au FileType pyrex set ai et nu sw=4 ts=4 tw=79 omnifunc=pythoncomplete#Complete
au FileType rst set ai et nu sw=2 ts=2
au FileType tex set ai et nu sw=2 ts=2
au FileType c set ai noet nu sw=8 ts=8
au FileType cpp set ai noet nu sw=8 ts=8
au FileType cuda set ai noet nu sw=8 ts=8
au FileType sh set ai et nu sw=2 ts=2 tw=79
au FileType csh set ai et nu sw=2 ts=2 tw=79
au FileType javascript set ai et nu sw=4 ts=4
au FileType html set ai et nu sw=4 ts=4
au FileType htmldjango set ai et nu sw=4 ts=4
au FileType css set ai et nu sw=4 ts=4
au FileType ruby set ai et nu sw=4 ts=4 tw=79
au FileType fortran set ai et nu sw=3 ts=3 tw=72
au FileType vim set ai nu
au BufRead *.spg set ft=fortran
au BufRead *.wiki set ai et nu ft=creole

" Make p in Visual mode replace the selected text with the "" register.
vnoremap p <Esc>:let current_reg = @"<CR>gvdi<C-R>=current_reg<CR><Esc>

" determine arrow for visual lines or text lines based on wrap.
function MapArrows ()
  setlocal wrap linebreak nolist
  set virtualedit=
  setlocal display+=lastline
  noremap  <buffer> <silent> <Up>   gk
  noremap  <buffer> <silent> <Down> gj
  noremap  <buffer> <silent> <Home> g<Home>
  noremap  <buffer> <silent> <End>  g<End>
  inoremap <buffer> <silent> <Up>   <C-o>gk
  inoremap <buffer> <silent> <Down> <C-o>gj
  inoremap <buffer> <silent> <Home> <C-o>g<Home>
  inoremap <buffer> <silent> <End>  <C-o>g<End>
endfunction " MapArrows
function UnMapArrows ()
  setlocal nowrap
  set virtualedit=all
  silent! nunmap <buffer> <Up>
  silent! nunmap <buffer> <Down>
  silent! nunmap <buffer> <Home>
  silent! nunmap <buffer> <End>
  silent! iunmap <buffer> <Up>
  silent! iunmap <buffer> <Down>
  silent! iunmap <buffer> <Home>
  silent! iunmap <buffer> <End>
endfunction " UnMapArrows
if &wrap
  call MapArrows()
else
  call UnMapArrows()
endif
function ToggleWrap()
  if &wrap
    echo "Wrap OFF"
    call UnMapArrows()
  else
    echo "Wrap ON"
    call MapArrows()
  endif
endfunction
noremap <silent> <Leader>w :call ToggleWrap()<CR>

function MyTabLine()
  let s = '' " complete tabline goes here
  " loop through each tab page
  for t in range(tabpagenr('$'))
    " switch selecting format based on tab number
    if t + 1 == tabpagenr()
      let s .= '%#TabLineSel#'
    else
      let s .= '%#TabLine#'
    endif
    " set the tab page number (for mouse clicks)
    let s .= '%' . (t + 1) . 'T'
    " set page number string
    let s .= t + 1 . ':'
    " get buffer names and statuses
    let n = '' " temp string for buffer names while we loop and check buftype
    let m = 0  " &modified counter
    let bc = len(tabpagebuflist(t + 1))  "counter to avoid last ' '
    " loop through each buffer in a tab
    for b in tabpagebuflist(t + 1)
      " buffer types: quickfix gets a [Q], help gets [H]{base fname}
      " others get 1dir/2dir/3dir/fname shortened to 1/2/3/fname
      if getbufvar( b, "&buftype" ) == 'help'
        let n .= '[H]' . fnamemodify( bufname(b), ':t:s/.txt$//' )
      elseif getbufvar( b, "&buftype" ) == 'quickfix'
        let n .= '[Q]'
      else
        let n .= pathshorten(bufname(b))
      endif
      " check and ++ tab's &modified count
      if getbufvar( b, "&modified" )
        let m += 1
      endif
      " no final ' ' added...formatting looks better done later
      if bc > 1
        let n .= ' '
      endif
      let bc -= 1
    endfor
    " add buffer names
    let s .= n
    " switch to non-selecting format
    let s .= ' %#TabLine#'
    " add modified label [n+] where n pages in tab are modified
    if m > 0
      let s .= '[' . m . '+] '
    endif
  endfor
  " after the last tab fill with TabLineFill and reset tab page nr
  let s .= '%#TabLineFill#%T'
  " right-align the label to close the current tab page
  if tabpagenr('$') > 1
    let s .= '%=%#TabLineFill#%999X[X]'
  endif
  return s
endfunction " MyTabLine
set stal=2
set tabline=%!MyTabLine()

" tab switch
function MapTabSwitch ()
  noremap <C-S-Tab> :tabprev<CR>
  noremap <C-Tab> :tabnext<CR>
  inoremap <C-S-Tab> <C-o>:tabprev<CR>
  inoremap <C-Tab> <C-o>:tabnext<CR>
endfunction " MapTabSwitch
call MapTabSwitch()

function EnableTerminalMeta (values)
  let ascii_nums = map(a:values, 'nr2char(v:val)')
  for char in ascii_nums
    exe "set <M-".char.">=\<Esc>".char
  endfor
endfunction " EnableTerminalMeta

function MapTabNumber (keybegin, keyend)
  let ascii_nums = range(49, 57) " skip zero.
  let ascii_nums = map(ascii_nums, 'nr2char(v:val)')
  for char in ascii_nums
    exe "noremap <silent> ".a:keybegin.char.a:keyend." :tabn ".char."<CR>"
  endfor
  exe "noremap <silent> ".a:keybegin."0".a:keyend." :tablast<CR>"
endfunction " MapTabNumber

if !has("gui")
  call EnableTerminalMeta(range(48,57))
  call MapTabNumber("<Esc>", "")
else
  if has("gui_macvim")
    set guitablabel=%N:%t\ %M
    call MapTabNumber("<D-", ">")
  else
    set guitablabel=%f
    call MapTabNumber("<M-", ">")
  endif
endif

" taglist
noremap <F4> :TlistToggle<cr>

" vundle
if filereadable(expand('~/.vim/bundle/Vundle.vim'))
  " set the runtime path to include Vundle and initialize
  set tags+=$HOME/.vim/tags/python.ctags
  set rtp+=~/.vim/bundle/Vundle.vim
  call vundle#begin()
  " alternatively, pass a path where Vundle should install plugins
  "call vundle#begin('~/some/path/here')

  " let Vundle manage Vundle, required
  Plugin 'gmarik/Vundle.vim'

  Plugin 'JuliaLang/julia-vim'

  " All of your Plugins must be added before the following line
  call vundle#end()            " required
  filetype plugin indent on    " required
endif

let g:showmarks_enable=0
let python_highlight_builtins = 1
let python_highlight_all = 1
" minibuf
let g:miniBufExplMapWindowNavVim = 1
let g:miniBufExplMapWindowNavArrows = 1
let g:miniBufExplMapCTabSwitchBufs = 1
let g:miniBufExplModSelTarget = 1 
" vim-latex
let g:tex_flavor='latex'

if filereadable(expand('~/self/etc/dot_vimrc'))
  so ~/self/etc/dot_vimrc
endif

if filereadable(expand('~/.vimrc_local'))
  so ~/.vimrc_local
endif
