from django.shortcuts import render
# from sklearn.externals import joblib
from .machine import inference, load_model
from django.views import View
from django.http import HttpResponse,JsonResponse, HttpResponseNotFound
import joblib
import logging

    
        
def predict_data(request):
    if request.method == 'POST':
        vectoriser, model = joblib.load('models/vectoriser.pickle'), joblib.load('models/Sentiment-LR.pickle')
        logging.info('Model Loaded Successfully')
        # vectoriser, model = load_model('models/vectoriser.pickle', 'models/Sentiment-LR.pickle')
        #  = load_model('/Users/sarwars/Desktop/Projects/sentiment_ana_django/models/Sentiment-LR.pickle')
        tweets = request.POST.get('comment')
        cols = ["tweet"]
        result_df = inference(vectoriser, model, tweets, cols)
        print(result_df)
        return JsonResponse({'sentiment': result_df, 'text': tweets})
    return render(request, 'prediction/predict.html')
        
        
class PredictData(View):
    """ receive the 'data' from API using method: 'POST'.
    body parameter, "comment: 'string'"
    comment will be processed 'inference()' method, after that
    response will be given in Json Form.
    """
    def get(self, request):
        return render(request, 'prediction/predict.html')
    
    def post(self, request):
        # print(request.POST)
        try:
            tweets = str(request.POST.get('comment', None))
            print(tweets, 'tweets')
            if tweets is None or tweets == '':
                print('Blank string is not acceptable')
                dict_msg={
                    'error': '404 Not Found',
                    'msg': 'Empty string is not acceptable'
                }
                return HttpResponseNotFound(JsonResponse(dict_msg))
                # return HttpResponseNotFound(JsonResponse(dict_msg))
            else:
                vectoriser, model = joblib.load('models/vectoriser.pickle'), joblib.load('models/Sentiment-LR.pickle')
                # vectoriser, model = load_model('/Users/sarwars/Desktop/Projects/sentiment_ana_django/models/vectoriser.pickle', '/Users/sarwars/Desktop/Projects/sentiment_ana_django/models/Sentiment-LR.pickle')
                cols = ["tweet"]
                result_df = inference(vectoriser, model, tweets, cols)
                resp = HttpResponse.status_code
                print(f'response: {resp}')
                return JsonResponse({'sentiment': result_df}, safe=False)
        except Exception as e:
            dict_msg={
                'error': 'True',
                'message': f'Problem on server side, {str(e)}',
            }
            return JsonResponse(dict_msg)