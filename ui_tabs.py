"""
Closable tabs in the main window's notebook (user interface only).

Every view a pipeline stage opens (the cell detection view, the signals &
correlation analysis, a correlation matrix, the 3D view, ...) is a tab of
the main window rather than a separate window. `add_tab` makes one: a
frame with a thin bar at the top, where the view can put its own menus
(left) next to a "Close Tab" button (right), and a body below it.

Without a notebook (e.g. a view used from a script) the same layout is
put in a separate window instead, so the views work either way.
"""

import tkinter as tk

BAR_BG = '#f5f5f5'
BAR_ACTIVE = '#dbeafe'
BAR_BORDER = '#e2e8f0'


class Tab:
    """A view's container: `frame` (the tab, or the window), `bar` (menus
    go here, packed side='left') and `body` (the view's content)."""

    def __init__(self, frame, bar, body, notebook):
        self.frame = frame
        self.bar = bar
        self.body = body
        self.notebook = notebook

    @property
    def is_tab(self):
        return self.notebook is not None

    def exists(self):
        try:
            return bool(self.frame.winfo_exists())
        except tk.TclError:
            return False

    def set_title(self, title):
        if not self.exists():
            return
        if self.is_tab:
            self.notebook.tab(self.frame, text=f"  {title}  ")
        else:
            self.frame.title(title)

    def show(self):
        """Bring the view to the front (select its tab / raise its window)."""
        if not self.exists():
            return
        if self.is_tab:
            if str(self.frame) in self.notebook.tabs() and self.notebook.tab(self.frame, 'state') == 'hidden':
                self.notebook.tab(self.frame, state='normal')
            self.notebook.select(self.frame)
        else:
            self.frame.deiconify()
            self.frame.lift()
            self.frame.focus_force()

    def close(self):
        """Remove the tab (or close the window); destroying a notebook's
        child also removes its tab."""
        if self.exists():
            self.frame.destroy()

    def add_menu(self, label, build):
        """A drop-down menu in the bar: build(menu) fills it."""
        mb = tk.Menubutton(self.bar, text=label, font=('Segoe UI', 9), bg=BAR_BG,
                           activebackground=BAR_ACTIVE, relief='flat', bd=0, padx=8, pady=3)
        menu = tk.Menu(mb, tearoff=0, font=('Segoe UI', 9))
        build(menu)
        mb['menu'] = menu
        mb.pack(side='left')
        return menu


def add_tab(notebook, title, parent=None, after=None, select=True, hidden=False, on_close=None,
            geometry="1200x800"):
    """Create a view: a tab of `notebook` (placed right after the tab
    `after`, if given, else at the end), or, when notebook is None, a
    window over `parent`. `on_close()` runs instead of just closing when
    the Close Tab button (or the window's close box) is used. `hidden`:
    the tab is added hidden (shown by Tab.show(), e.g. once its content
    has loaded). Returns a Tab."""
    if notebook is not None:
        frame = tk.Frame(notebook)
        index = 'end'
        if after is not None:
            try:
                index = notebook.index(after) + 1
            except tk.TclError:
                index = 'end'
        if index != 'end' and index >= notebook.index('end'):
            index = 'end'
        if index == 'end':
            notebook.add(frame, text=f"  {title}  ")
        else:
            notebook.insert(index, frame, text=f"  {title}  ")
        if hidden:
            notebook.tab(frame, state='hidden')
    else:
        frame = tk.Toplevel(parent)
        frame.title(title)
        frame.geometry(geometry)

    bar = tk.Frame(frame, bg=BAR_BG, height=26, highlightbackground=BAR_BORDER, highlightthickness=1)
    bar.pack(side='top', fill='x')
    bar.pack_propagate(False)
    body = tk.Frame(frame)
    body.pack(side='top', fill='both', expand=True)
    tab = Tab(frame, bar, body, notebook)

    close = on_close or tab.close
    tk.Button(bar, text="✕ Close Tab" if notebook is not None else "✕ Close", command=close,
              font=('Segoe UI', 9), bg=BAR_BG, activebackground=BAR_ACTIVE, relief='flat', bd=0,
              padx=8).pack(side='right')
    if notebook is None:
        frame.protocol("WM_DELETE_WINDOW", close)
    if select and not hidden:
        tab.show()
    return tab
