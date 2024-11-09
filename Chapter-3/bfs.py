graph={
    '5' : ['3','7'],
    '3' : ['2','4'],
    '7' :['8'],
    '2' :[],
    '4' :['8'],
    '8' :[]
}
visited=[]
queue=[]
  
def bfs(visited,graph,node): #funtion for bfs
    visited.append(node)
    queue.append(node)
    while queue:
        m=queue.pop(0) #creating loop
        print(m,end=" ")
        for neighbour in graph[m]:
            if  neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


#driver code
print("following the bfs")
bfs(visited,graph,'5')  #function calling