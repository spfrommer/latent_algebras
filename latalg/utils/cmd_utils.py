from __future__ import annotations

import dacite

from typing import Any, Dict


def add_click_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def get_params_loader(all_params: Dict[str, Any]):
    ignore_types_config = dacite.Config(check_types=False)
    return lambda data_class: dacite.from_dict(
        data_class=data_class, data=all_params, config=ignore_types_config,
    )