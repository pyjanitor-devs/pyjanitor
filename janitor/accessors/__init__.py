"""Top-level imports for pyjanitor's dataframe accessors."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "data_description": ["DataDescription"],
    },
)
