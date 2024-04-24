class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == 'DropNode':
            x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)
            x = x / (1 - drop_rate)
        elif self.dropping_method == 'DropEdge':
            edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
            if isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
                edge_index = torch.stack((row, col))
            perm = torch.randperm(edge_index.size(1))
            edge_index = edge_index.t()[perm][:edge_reserved_size].t()  # 打乱后取前edge_reserved_size个

        return x, edge_index

    def getNewWeight(self, add_self_loops, normalize, x: Tensor, edge_index: Adj):
        # add self loop
        if add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # normalize
        edge_weight = None
        if normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight


class MDGCL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MSGCF2, self).__init__(config, dataset)
        # load parameters info
        self.latent_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.layer_cl = config['layer_cl']
        self.drop_ratio = config["drop_ratio"]  # 新增的配置
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.dropping_method = config["dropping_method"]
        self.temperature = config['temperature']
        self.dataset = dataset
        self.mf_loss = BPRLoss()
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        message_drop = 0
        if self.dropping_method == 'DropMessage':
            message_drop = self.drop_ratio
        self.gcn_conv = MessageDropoutLightGCNConv(dim=self.latent_dim, drop_rate=message_drop)
        self.reg_loss = EmbLoss()
        self.mf_loss = BPRLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.drop_block = DropBlock(self.dropping_method)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, drop=False):
        all_embeddings = self.get_ego_embeddings()
        all_embs_cl = all_embeddings
        embeddings_list = [all_embeddings]
        if not drop:
            for layer_idx in range(self.n_layers):
                all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
                embeddings_list.append(all_embeddings)
        else:
            for layer_idx in range(self.n_layers):
                x = all_embeddings
                edge_index = self.edge_index
                edge_weight = self.edge_weight
                if self.training and (self.dropping_method == 'DropEdge' or self.dropping_method == 'DropNode'):
                    x, edge_index = self.drop_block.drop(all_embeddings, edge_index, self.drop_ratio)
                    if self.dropping_method == 'DropEdge':
                        edge_index, edge_weight = self.drop_block.getNewWeight(True, True, x, edge_index)
                all_embeddings = self.gcn_conv(x, edge_index, edge_weight, True)
                embeddings_list.append(all_embeddings)
                if layer_idx == self.layer_cl - 1:
                    all_embs_cl = all_embeddings
        embeddings_list = torch.stack(embeddings_list, dim=1)
        embeddings_list = torch.mean(embeddings_list, dim=1, keepdim=False)
        user_all_embeddings, item_all_embeddings = torch.split(embeddings_list, [self.n_users, self.n_items], dim=0)
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.n_users, self.n_items])
        if drop:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        # bpr loss
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)
        mf_loss = self.mf_loss(p_scores, n_scores)

        # reg loss
        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        reg_loss = self.reg_loss(u_e_p, pi_e_p, ni_e_p)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        # 只用一次forward
        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl = self.forward(
            drop=True)
        u_embeddings = user_all_embeddings[user_list]
        pos_embeddings = item_all_embeddings[pos_item_list]
        neg_embeddings = item_all_embeddings[neg_item_list]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user_list)
        pos_ego_embeddings = self.item_embedding(pos_item_list)
        neg_ego_embeddings = self.item_embedding(neg_item_list)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        # calculate CL Loss
        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])
        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[user], user_all_embeddings_cl[user])
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[pos_item], item_all_embeddings_cl[pos_item])

        return mf_loss + self.reg_weight * reg_loss, self.ssl_weight * (user_cl_loss + item_cl_loss)

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
