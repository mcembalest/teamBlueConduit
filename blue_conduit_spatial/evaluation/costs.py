import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from time import time 
from blue_conduit_spatial.utilities import load_predictions
from blue_conduit_spatial.evaluation import generate_hit_rate_curve_by_partition

class CostsHandler:
    
    def __init__(self, Ydata, train_pid, test_pid, partitions_builder, pred_dir, models_prefix, compute_savings=True, 
                 bl_prefix='baseline'):
        self.Ydata = Ydata
        self.train_pid = train_pid
        self.test_pid = test_pid
        self.partitions_builder = partitions_builder
        self.models_prefix = models_prefix
        self.models_preds = self.load_preds(pred_dir)
        self.compute_savings = compute_savings
        self.dig_data = {}
        self.hexagons = {} # hexagon objects indexed by hexagon resolution
        self.costs = {} # pooled costs dictionary indexed by ts, resolution and strategy
        self.bl_prefix = bl_prefix
        
    def load_preds(self, pred_dir):
        models_preds = {}
        for model_prefix_i in self.models_prefix:
            train_pred_all_i, test_pred_all_i = load_predictions(pred_dir, probs_prefix=model_prefix_i)
            models_preds[model_prefix_i] = {}
            models_preds[model_prefix_i]['train_pred_all'] = train_pred_all_i
            models_preds[model_prefix_i]['test_pred_all'] = test_pred_all_i
        return models_preds

    def dig_data_ts_res(self, ts, res):
        '''
        Retrieve digging data per hit_rate curve for one selection of train size and resolution,
        for all strategies.
        '''
        
        t0 = time()
        hexagons_res = self.hexagons[res]
        parcel_gdf = hexagons_res.parcel_gdf
        dig_data = dict([(strategy, {}) for strategy in self.models_prefix])
        
        self.splits = list(range(len(self.test_pid[f'ts_{ts}'][f'res_{res}'])))
        for strategy in self.models_prefix:
            for split_ in self.splits:
                test_pid_i = self.test_pid[f'ts_{ts}'][f'res_{res}'][split_]
                y_true_i = self.Ydata.loc[test_pid_i]['dangerous'].values.astype('float')
                y_pred_i = self.models_preds[strategy]['test_pred_all'][f'ts_{ts}'][f'res_{res}'][split_]

                hr_args_i = {
                    'parcel_df':parcel_gdf,
                    'threshold_init':0.9,
                    'gen_dig_metadata': True,
                    'pid_list': test_pid_i,
                    'y_true': y_true_i,
                    'y_pred': y_pred_i
                }
                _, _, data_strategy_split = generate_hit_rate_curve_by_partition(**hr_args_i)
                dig_data[strategy][split_] = data_strategy_split

        t1 = time()
        msg = f'Digging data computation for ts:{ts}, n_hexagons:{res} done. | Total time: {t1-t0:.2f} s.'
        print(msg)

        return dig_data
    
    def compute_dig_data(self, ts_list, res_list):
        '''
        Get lead digging process data for each strategy in self.models_prefix
        for all train sizes in `ts_list` and resolutions in `res_list`
        '''
        for ts in ts_list:
            for res in res_list:
                if f'ts_{ts}' in self.dig_data:
                    if f'res_{res}' in self.dig_data[f'ts_{ts}']:
                        continue
                else:
                    self.dig_data[f'ts_{ts}'] = {}
                result_ts_res = self.dig_data_ts_res(ts, res)
                self.dig_data[f'ts_{ts}'][f'res_{res}'] = result_ts_res
        
        return self.dig_data

    def cost_curve(self, dig_data_ts_res_strategy):
        '''
        Generate cost curve per split for one selection of:
            - strategy
            - hexagon resolution 
            - train size

        Inputs
        ----------
        dig_data_ts_res_strategy: dict
            Dictionary indexed by split index and values as `generate_hit_rate_curve_by_partition` outputs
            for the given strategy, hexagon resolution and train size.

        Return
        ----------

        costs: dict
            Dictionary with split index keys and cost curve pd.DataFrame as values
        '''
        
        agg_fn = {'dig_threshold': ['min'], 'dig_index':['min']}
        cols_map = {'dig_threshold_min': 'prob_thres', 
                    'dig_index_min': 'digs'}

        costs = {}
        for split_ in self.splits:
            costs_split = dig_data_ts_res_strategy[split_].sort_values('dig_index')
            costs_split['lead_digs'] = costs_split['true_val'].cumsum()
            costs_split['dig_index'] += 1
            costs_split = costs_split.groupby('lead_digs').agg(agg_fn)
            costs_split.columns = ['_'.join(col) for col in costs_split.columns.values]
            costs_split = costs_split.rename(columns=cols_map)
            costs_split = costs_split.reset_index()
            costs_split['cost'] = 5000*costs_split['lead_digs'] + 3000*(costs_split['digs']-costs_split['lead_digs'])
            costs_split['cost_avg'] = costs_split['cost']/costs_split['lead_digs']
            costs_split = costs_split[~(costs_split['lead_digs']==0)]
            costs[split_] = costs_split
        return costs
    
    def add_savings(self, costs_strategy, costs_baseline):
        for split_ in self.splits:
            # Average savings per lead piepe until the i-th digged lead pipe
            costs_strategy[split_]['savings_avg'] = costs_baseline[split_]['cost_avg'].values-costs_strategy[split_]['cost_avg'].values    
            # Cumulated savings until the i-th digged lead pipe
            costs_strategy[split_]['savings'] = costs_baseline[split_]['cost'].values-costs_strategy[split_]['cost'].values    
        return costs_strategy

    def pooling_splits(self, costs_strategy, bl_bool=False):
        # Pooled savings and splits across splits for a single strategy, hexagon resolution
        # and train size
        min_leads = min([costs.shape[0] for costs in costs_strategy.values()])
        lead_digs = range(1,min_leads+1)
        pooled_cost = np.mean([costs.iloc[:min_leads+1].cost for costs in costs_strategy.values()])
        pooled_cost_avg = np.mean([costs.iloc[:min_leads+1].cost_avg for costs in costs_strategy.values()])
        result = {'cost':pooled_cost, 
                  'cost_avg':pooled_cost_avg}

        if self.compute_savings and not bl_bool:
            pooled_savings = np.mean([costs.iloc[:min_leads+1].savings for costs in costs_strategy.values()])
            pooled_savings_avg = np.mean([costs.iloc[:min_leads+1].savings_avg for costs in costs_strategy.values()])
            result['savings']=pooled_savings 
            result['savings_avg']=pooled_savings_avg

        return result

    def pooled_cost_curve(self, dig_data_ts_res):
        '''
        Cost savings for all strategies for one train size and hexagon resolution, 
        pooled across all splits per strategy.
        '''
        # Pooled results across splits per strategy
        pooled_costs = {}

        # Calculate baseline costs for savings calculation
        costs_baseline = self.cost_curve(dig_data_ts_res[self.bl_prefix])
        pooled_costs_bl = self.pooling_splits(costs_baseline, bl_bool=True)
        pooled_costs[self.bl_prefix] = pooled_costs_bl

        for strategy in self.models_prefix:
            costs_strategy = {}
            if strategy==self.bl_prefix:
                continue
            df_ts_res_strategy = dig_data_ts_res[strategy]
            costs_strategy = self.cost_curve(df_ts_res_strategy)
            if self.compute_savings:
                costs_strategy = self.add_savings(costs_strategy, costs_baseline)
            pooled_costs[strategy] = self.pooling_splits(costs_strategy)

        return pooled_costs
    
    def load_hexagons(self, res_list):
        # If hexagons for the given resolutions haven't been loaded yet, load them
        for res in res_list:
            if res in self.hexagons:
                continue
            self.hexagons[res] = self.partitions_builder.Partition(partition_type='hexagon', num_cells_across=res)

    def compute_costs(self, ts_list, res_list):
        
        # If hexagons haven't been loaded or digging data hasn't been computed for the given train sizes and 
        # hexagons resolutions, then load/compute them.
        self.load_hexagons(res_list)
        self.compute_dig_data(ts_list, res_list)
            
        ## Set up costs dictionary if it hadn't been initialized yet
            
        for ts in ts_list:
            for res in res_list:
                # If the costs for ts and res have been calculated already, skip
                if f'ts_{ts}' in self.costs:
                    if f'res_{res}' in self.costs[f'ts_{ts}']:
                        continue
                else:
                    self.costs[f'ts_{ts}'] = {}
                
                self.splits = list(range(len(self.test_pid[f'ts_{ts}'][f'res_{res}'])))
                
                t0 = time()
                dig_data_ts_res = self.dig_data[f'ts_{ts}'][f'res_{res}']
                costs_ts_res = self.pooled_cost_curve(dig_data_ts_res)
                self.costs[f'ts_{ts}'][f'res_{res}'] = costs_ts_res

                t1 = time()
                msg = f'Costs computation for ts:{ts}, n_hexagons:{res} done. | Total time: {t1-t0:.2f} s.'
                print(msg)
        
        return self.costs

    def plot_costs(self, res, ts, savefig=False, norm_x=True, zoom_perc=0.9, plot_dir=None, metric='savings'):
        # If costs for this resolutions and train size haven't been calculated yet, do so.
        costs = None
        if f'ts_{ts}' in self.dig_data:
            if f'res_{res}' in self.dig_data[f'ts_{ts}']:
                costs = self.costs[f'ts_{ts}'][f'res_{res}']
        
        if costs is None:
            self.compute_costs([ts], [res])
            costs = self.costs[f'ts_{ts}'][f'res_{res}']
                
        N = len(list(costs.values())[0]['cost'])
        zoom = int(zoom_perc*N)
        cmap = sns.color_palette()

        fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=100)
        for i, strategy in enumerate(self.models_prefix):
            if metric=='savings' and strategy=='baseline':
                continue
            color = cmap[i]
            x = np.array(list(range(1,zoom+1)))
            if norm_x:
                x = 100*x/N

            y = costs[strategy][f'{metric}'][:zoom]
            lowess = sm.nonparametric.lowess(y, x, frac=0.05) # smooth the curve
            ax[0].plot(x, y, color=color, alpha=0.15)
            ax[0].plot(lowess[:, 0], lowess[:, 1], color=color, label=strategy)

            y_avg = costs[strategy][f'{metric}_avg'][:zoom]
            lowess = sm.nonparametric.lowess(y_avg, x, frac=0.05) # smooth the curve
            ax[1].plot(x, y_avg, color=color, alpha=0.15)
            ax[1].plot(lowess[:, 0], lowess[:, 1], color=color, label=strategy)

        if norm_x:
            fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
            xticks = mtick.FormatStrFormatter(fmt)
            ax[0].xaxis.set_major_formatter(xticks)
            ax[0].set_xlabel('Share of removed lead pipes (%)')
            ax[1].xaxis.set_major_formatter(xticks)
            ax[1].set_xlabel('Share of removed lead pipes (%)')
        
        ax[0].set_ylabel(f'Cumulative {metric} ($USD)')
        ax[0].yaxis.set_major_formatter('${x:,.2f}')
        
        ax[1].set_ylabel(f'Average {metric}/pipe ($USD)')
        ax[1].yaxis.set_major_formatter('${x:,.2f}')
        
        if metric=='savings':
            ax[0].axhline(0, color='k', lw=0.5)
            ax[1].axhline(0, color='k', lw=0.5)
            ax[0].set_ylim(-abs(4e5),abs(4e5))
            ax[1].set_ylim(-abs(100),abs(100))
            
        lgd = ax[1].legend(loc='upper center', bbox_to_anchor=(-0.15, -0.15),
                  fancybox=True, shadow=False, ncol=4)
        
        if metric=='savings':
            title = f'Savings over Baseline strategy\nHexagon resolution:{res} | Train size:{ts}'
        else:
            title = f'Costs per strategy\nHexagon resolution:{res} | Train size:{ts}'
        title = fig.suptitle(title, y=1.05)
        fig.subplots_adjust(wspace=0.3)
        if savefig:
            if norm_x:
                savepath = f'{plot_dir}/{metric}_norm_ts_{ts}_n_hex_{res}.png'
            else:
                savepath = f'{plot_dir}/{metric}_ts_{ts}_n_hex_{res}.png'
            plt.savefig(savepath, dpi=150, bbox_extra_artists=(lgd,title), bbox_inches='tight')
        plt.show()