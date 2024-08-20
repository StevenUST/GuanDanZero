from guandan_net_tensorflow import GuandanNetForTwo

m_path = "./saved_model/guandan_model_v1_0.model.data-00000-of-00001"

if __name__ == "__main__":
    model = GuandanNetForTwo()
    model.restore_model(m_path)