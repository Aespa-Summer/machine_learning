#include<bits/stdc++.h>
using namespace std;
using i64 = long long;
void solve(){
	i64 n,x,f=0,mn = 100000000000000000;
	vector<i64> a(100010);
	cin>>n;
	for(int i =1;i<n+1;++i){
		cin>>a[i];
	}
	for(int i =1;i<n+1;++i){
		x = i;
		f = 0;
		for(int i =1;i<n+1;++i){
			f += pow(x-i,2)*a[i];
		}
		if(f<mn){
			mn = f;
		}
	}
	cout<<mn<<endl;
}
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int t;
	//cin >> t;
	t = 1;
	while(t--){
		solve();
	}
	return 0;
}



