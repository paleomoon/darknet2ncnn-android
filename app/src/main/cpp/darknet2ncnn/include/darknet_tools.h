/**
 * @File   : darknet_tools.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/30/2018, 9:27:56 AM
 */

#ifndef _DARKNET_TOOLS_H
#define _DARKNET_TOOLS_H

#include <darknet.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>

static std::string get_layer_output_blob_format(layer l)
{
  LAYER_TYPE lt = l.type;
  if (lt == CONVOLUTIONAL)
  {
    return "conv_%d_activation";
  }
  else if (lt == DECONVOLUTIONAL)
  {
    return "deconv_%d_activation";
  }
  else if (lt == LOCAL)
  {
    return "local_%d_activation";
  }
  else if (lt == ACTIVE)
  {
    return "activation_%d";
  }
  else if (lt == LOGXENT)
  {
    return "activation_%d";
  }
  else if (lt == L2NORM)
  {
    return "l2norm_%d";
  }
  else if (lt == RNN)
  {
    return "rnn_%d";
  }
  else if (lt == GRU)
  {
    return "gru_%d";
  }
  else if (lt == LSTM)
  {
    return "lstm_%d";
  }
  else if (lt == CRNN)
  {
    return "crnn_%d";
  }
  else if (lt == CONNECTED)
  {
    return "connected_%d_activation";
  }
  else if (lt == CROP)
  {
    return "crop_%d";
  }
  else if (lt == COST)
  {
    return "cost_%d";
  }
  else if (lt == REGION)
  {
    return "region_%d";
  }
  else if (lt == YOLO)
  {
    return "yolo_%d";
  }
  else if (lt == DETECTION)
  {
    return "detection_%d";
  }
  else if (lt == SOFTMAX)
  {
    return "softmax_%d";
  }
  else if (lt == NORMALIZATION)
  {
    return "normalization_%d";
  }
  else if (lt == BATCHNORM)
  {
    return "batch_norm_%d";
  }
  else if (lt == MAXPOOL)
  {
    return "maxpool_%d";
  }
  else if (lt == REORG)
  {
    return "reorg_%d";
  }
  else if (lt == AVGPOOL)
  {
    return "global_avg_pool_%d";
  }
  else if (lt == ROUTE)
  {
    return "route_%d";
  }
  else if (lt == UPSAMPLE)
  {
    return "upsample_%d";
  }
  else if (lt == SHORTCUT)
  {
    return "shortcut_%d_activation";
  }
  else if (lt == DROPOUT)
  {
    return "dropout_%d";
  }
  else
  {
    fprintf(stderr, "Type not recognized: %d\n", l.type);
    return "";
  }
  return "";
}

static std::string get_layer_no_split_output_blob_name(layer l, int layer_index)
{
  char names[64] = {0};
  sprintf(names, get_layer_output_blob_format(l).c_str(), layer_index);
  return names;
}

typedef std::map<int, std::string> splits_type;
struct split_data
{
  int layer_index;
  std::string output_blob_name;
  std::map<int, std::string> splits;

  void add_new_split_info(int input_layer_index)
  {
    char split_id[128] = {0};
    sprintf(split_id, "_split_%d", (int)splits.size());
    splits.insert(std::pair<int, std::string>(input_layer_index, split_id));
  }

  std::string get_split_info(int input_layer_index)
  {
    splits.find(input_layer_index);
    splits_type::iterator iter = splits.find(input_layer_index);
    if (iter != splits.end())
    {
      return iter->second;
    }
    return "";
  }

  std::string get_split_full_info(int input_layer_index)
  {
    return output_blob_name + get_split_info(input_layer_index);
  }
};

static std::string get_layer_output_blob_name(std::vector<split_data> &split_info, int input_index, int output_index)
{
  if (split_info.size() > input_index && split_info[input_index].splits.size() > 0)
  {
    split_data &split = split_info[input_index];
    if (split.splits.size() == 1)
    {
      return split.output_blob_name;
    }
    else
    {
      return split.get_split_full_info(output_index);
    }
  }
  return "";
}

static void get_layers_split_info(network net, std::vector<split_data> &split_info)
{
  split_info.clear();
  for (int n = 0; n < net.n; n++)
  {
    split_data split;
    split.layer_index = n;
    layer l = net.layers[n];
    split.output_blob_name = get_layer_no_split_output_blob_name(l, n);
    split_info.push_back(split);

    if (n == 0)
      continue;

    if (l.type == SHORTCUT)
    {
      split_info[n - 1].add_new_split_info(n);
      split_info[l.index].add_new_split_info(n);
      continue;
    }
    else if (l.type == ROUTE)
    {
      for (int i = 0; i < l.n; i++)
      {
        int index = l.input_layers[i];
        split_info[index].add_new_split_info(n);
      }
    }
    else
    {
      split_info[n - 1].add_new_split_info(n);
    }
  }
}

static image get_input_data_from_image(char *filename, int net_w,
                                       int net_h)
{
  image new_image = load_image_color(filename, 0, 0);
  image letter_image = letterbox_image(new_image, net_w, net_h);
  if (new_image.data != letter_image.data)
  {
    free_image(new_image);
  }
  return letter_image;
}

#endif