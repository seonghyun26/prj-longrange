from torch_geometric.datasets import GNNBenchmarkDataset
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.loader.dataset.graph2linegraph import graph2LineGraph

class GNNBenchmarkDataset_LG(GNNBenchmarkDataset):
  def process(self):
    if self.name == 'CSL':
            data_list = self.process_CSL()
            torch.save(self.collate(data_list), self.processed_paths[0])
    else:
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
        
            
                if hasattr(data_list[0], 'EigVals') == False:
                    print("Preprocessing LapPE of line graph...")
                    data = compute_posenc_stats(data, ['LapPE'], False, cfg)
                    print("Done...!")

            torch.save(self.collate(data_list), self.processed_paths[i])

