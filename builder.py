"""Graph builder from pandas dataframes"""
from collections import namedtuple

import dgl
import pandas as pd

from pandas.api.types import (
    # is_categorical,
    is_categorical_dtype,
    is_numeric_dtype,
)

__all__ = ["PandasGraphBuilder"]

"""
1、构建异构图，整体有一个PandasGraphBuilder类
2、包括添加实体的函数、添加实体关系的函数以及创建异构图的函数
3、根据data_dict和num_nodes_dict构建异构图
4、整体上就是，原始数据中有user侧的df_user，有item侧的df_item，以及user_item关系的df_relation
将这几种数据，加载到graph中
"""


def _series_to_tensor(series):
    if is_categorical_dtype(series):
        return torch.LongTensor(series.cat.codes.values.astype("int64"))
    else:  # numeric
        return torch.FloatTensor(series.values)


class PandasGraphBuilder(object):
    """Creates a heterogeneous graph from multiple pandas dataframes.

    Examples
    --------
    Let's say we have the following three pandas dataframes:

    User table ``users``:

    ===========  ===========  =======
    ``user_id``  ``country``  ``age``
    ===========  ===========  =======
    XYZZY        U.S.         25
    FOO          China        24
    BAR          China        23
    ===========  ===========  =======

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========

    # One could then create a bidirectional bipartite graph as follows:
    # >>> builder = PandasGraphBuilder()
    # >>> builder.add_entities(users, 'user_id', 'user')
    # >>> builder.add_entities(games, 'game_id', 'game')
    # >>> builder.add_binary_relations(plays, 'user_id', 'game_id', 'plays')
    # >>> builder.add_binary_relations(plays, 'game_id', 'user_id', 'played-by')
    # >>> g = builder.build()
    # >>> g.num_nodes('user')
    # 3
    # >>> g.num_edges('plays')
    # 4
    """

    def __init__(self):
        self.entity_tables = {}
        self.relation_tables = {}

        self.entity_pk_to_name = (
            {}
        )  # mapping from primary key name to entity name
        self.entity_pk = {}  # mapping from entity name to primary key
        self.entity_key_map = (
            {}
        )  # mapping from entity names to primary key values
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}  # mapping from relation name to source key
        self.relation_dst_key = (
            {}
        )  # mapping from relation name to destination key

    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype("category")
        if not (entities.value_counts() == 1).all():
            raise ValueError(
                "Different entity with the same primary key detected."
            )
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(
            entity_table[primary_key].values
        )

        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        self.entity_tables[name] = entity_table

    def add_binary_relations(
        self, relation_table, source_key, destination_key, name
    ):
        src = relation_table[source_key].astype("category")
        src = src.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[source_key]
            ].cat.categories
        )
        dst = relation_table[destination_key].astype("category")
        dst = dst.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[destination_key]
            ].cat.categories
        )
        if src.isnull().any():
            raise ValueError(
                "Some source entities in relation %s do not exist in entity %s."
                % (name, source_key)
            )
        if dst.isnull().any():
            raise ValueError(
                "Some destination entities in relation %s do not exist in entity %s."
                % (name, destination_key)
            )

        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (
            src.cat.codes.values.astype("int64"),
            dst.cat.codes.values.astype("int64"),
        )
        self.relation_tables[name] = relation_table
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        # Create heterograph
        # dgl.heterograph(data_dict, num_nodes_dict)
        # 其中data_dict可能是：data_dict = {
        #     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        #     ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
        #     ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
        # }
        # 而num_nodes_dict可能是：num_nodes_dict = {'user': 4, 'topic': 4, 'game': 6}
        #

        graph = dgl.heterograph(
            self.edges_per_relation, self.num_nodes_per_type
        )
        return graph


def make_demo_dataframe():
    """
    创建一个demo的数据集，用于测试类和函数的效果
        User table ``users``:

    ===========  ===========  =======
    ``user_id``  ``country``  ``age``
    ===========  ===========  =======
    XYZZY        U.S.         25
    FOO          China        24
    BAR          China        23
    ===========  ===========  =======

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========
    """
    # 创建user的dataframe
    user_id_list = ["XYZZY", "FOO", "BAR"]
    country_list = ["U.S.", "China", "China"]
    age_list = [25, 24, 23]
    user_dict = {"user_id": user_id_list, "country": country_list, "age": age_list}

    user = pd.DataFrame(user_dict)

    # 创建game的dataframe
    game_id_list = [1,2]
    title_list = ["Minecraft", "Teries 99"]
    is_sandbox_list = [True, False]
    is_multiplayer_list = [True, True]
    game_dict = {"game_id": game_id_list, "title": title_list, "is_sandbox": is_sandbox_list, "is_multiplayer": is_multiplayer_list}

    game = pd.DataFrame(game_dict)

    # 创建relation的dataframe
    user_id_list = ["XYZZY", "FOO", "FOO", "BAR"]
    game_id_list = [1, 1, 2, 2]
    hours_list = [24, 20, 16, 28]
    relation_dict = {"user_id": user_id_list, "game_id":game_id_list, "hours": hours_list}

    relation = pd.DataFrame(relation_dict)

    return user, game, relation


    # >>> builder = PandasGraphBuilder()
    # >>> builder.add_entities(users, 'user_id', 'user')
    # >>> builder.add_entities(games, 'game_id', 'game')
    # >>> builder.add_binary_relations(plays, 'user_id', 'game_id', 'plays')
    # >>> builder.add_binary_relations(plays, 'game_id', 'user_id', 'played-by')
    # >>> g = builder.build()
    # >>> g.num_nodes('user')
    # 3
    # >>> g.num_edges('plays')
    # 4
    # """
if __name__ == "__main__":
    users, games, relation = make_demo_dataframe()
    builder = PandasGraphBuilder()
    builder.add_entities(users, "user_id", "user")
    builder.add_entities(games, "game_id", "game")
    builder.add_binary_relations(relation, 'user_id', 'game_id', 'plays')
    builder.add_binary_relations(relation, 'game_id', 'user_id', 'played-by')
    g = builder.build()
    print(f"the num nodes is: {g.num_nodes('user')}")
    print(f"the num edges is: {g.num_edges('plays')}")
