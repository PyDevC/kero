#pragma once

#include "common.h"
#include <string>
#include <vector>

#include <execution_kernels/LogicPrimitives.h>
#include <execution_graph/graph.h>

std::string runGeneratePhysicalGraph(uint32_t masterIndex,
                                     std::vector<std::string> worker_ids,
                                     std::string query);

std::shared_ptr<ral::cache::graph> runGenerateGraph(uint32_t masterIndex,
	std::vector<std::string> worker_ids,
	std::vector<std::string> tableNames,
	std::vector<TableSchema> tableSchemas,
	std::string query,
	std::vector<std::vector<std::map<std::string, std::string>>> uri_values,
	std::map<std::string, std::string> config_options,
	std::string sql,
	std::string current_timestamp);

void startExecuteGraph(std::shared_ptr<ral::cache::graph> graph);
std::unique_ptr<PartitionedResultSet> getExecuteGraphResult(std::shared_ptr<ral::cache::graph> graph);

TableScanInfo getTableScanInfo(std::string logicalPlan);

std::unique_ptr<ResultSet> runSkipData(
	ral::frame::BlazingTableView metadata,
	std::vector<std::string> all_column_names,
	std::string query);


extern "C" {
std::pair<std::unique_ptr<PartitionedResultSet>, error_code_t> runQuery_C(int32_t masterIndex,
	std::vector<std::string> worker_ids,
	std::vector<std::string> tableNames,
	std::vector<TableSchema> tableSchemas,
	std::string query,
	std::vector<std::vector<std::map<std::string, std::string>>> uri_values,
	std::map<std::string, std::string> config_options);

std::pair<TableScanInfo, error_code_t> getTableScanInfo_C(std::string logicalPlan);

std::pair<std::unique_ptr<ResultSet>, error_code_t> runSkipData_C(
	ral::frame::TableView metadata, 
	std::vector<std::string> all_column_names, 
	std::string query);
} 
