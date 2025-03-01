from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QProgressBar

from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QPushButton,
    QApplication,
    QDialog,
)
from omegaconf import OmegaConf, DictConfig
from typing import Union


def create_parameter_box(
    min_val: float, max_val: float, text: str, step: float = 1, callback=None
):
    layout = QHBoxLayout()

    dial = QSpinBox()
    dial.setMaximumHeight(28)
    dial.setMaximumWidth(150)
    dial.setMinimum(min_val)
    dial.setMaximum(max_val)
    dial.setAlignment(Qt.AlignmentFlag.AlignRight)
    dial.setSingleStep(step)
    dial.valueChanged.connect(callback)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(dial)

    return dial, layout


def create_gauge(text: str):
    layout = QHBoxLayout()

    gauge = QProgressBar()
    gauge.setMaximumHeight(28)
    gauge.setMaximumWidth(200)
    gauge.setAlignment(Qt.AlignmentFlag.AlignCenter)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(gauge)

    return gauge, layout


def apply_to_all_children_widget(layout, func):
    # deliberately non-recursive
    for i in range(layout.count()):
        func(layout.itemAt(i).widget())


# The Annotation ComboBox for phase state
class TreeComboBox(QComboBox):
    def __init__(
        self,
        content_dict: Union[DictConfig, list, dict],
        parent=None,
    ):
        super(TreeComboBox, self).__init__(parent)
        self.setEditable(True)  # 使QComboBox可编辑以显示自定义文本
        self.tree_widget = QTreeWidget()
        # 设置标题栏
        self.tree_widget.setHeaderLabels(["To Select list"])
        # self.tree_widget.setHeaderHidden(True)  # 隐藏标题栏

        if type(content_dict) == DictConfig:
            content_dict = OmegaConf.to_container(content_dict)

        self.add_tree_item(self.tree_widget, (content_dict))

        # # 添加示例数据
        # parent_item = QTreeWidgetItem(self.tree_widget, ["Parent 1"])
        # child_item = QTreeWidgetItem(parent_item, ["Child 1"])
        # child_item = QTreeWidgetItem(parent_item, ["Child 2"])

        # parent_item = QTreeWidgetItem(self.tree_widget, ["Parent 2"])
        # child_item = QTreeWidgetItem(parent_item, ["Child 3"])
        # child_item = QTreeWidgetItem(parent_item, ["Child 4"])

        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

    def add_tree_item(self, parent, content):

        if isinstance(content, dict):
            for key, value in content.items():
                parent_item = QTreeWidgetItem(parent, [key])
                self.add_tree_item(parent_item, value)

        elif isinstance(content, list):
            for item in content:
                child_item = QTreeWidgetItem(parent, [item])

    def reset_tree(self, content_dict: Union[DictConfig, list, dict]):
        self.tree_widget.clear()
        self.add_tree_item(self.tree_widget, (content_dict))

    def showPopup(self):
        dialog = QDialog(self)
        layout = QVBoxLayout()
        layout.addWidget(self.tree_widget)
        dialog.setLayout(layout)
        dialog.setWindowTitle("Select Item")
        dialog.exec_()

    # # 如果点击的是子项，显示父子项的信息
    # if item.parent():
    #     parent_text = item.parent().text(0)
    #     child_text = item.text(0)
    #     self.setCurrentText(f"{parent_text}: {child_text}")
    # else:
    #     # 如果点击的是父项，显示父项的信息
    #     self.setCurrentText(item.text(0))
    # # 确保关闭对话框并显示选择的文本
    # self.tree_widget.parent().close()

    def get_text_tree(self, item, text=""):
        if text != "":
            text = item.text(0) + ":" + text
        else:
            text = item.text(0)

        if item.parent():
            return self.get_text_tree(item.parent(), text)
        else:
            return text

    def on_tree_item_clicked(self, item):
        # 这个函数我希望递归实现，如果点击的是子项，显示父子项的信息，如果点击的是父项，显示父项的信息
        self.setCurrentText(self.get_text_tree(item))
        # 确保关闭对话框并显示选择的文本
        self.tree_widget.parent().close()
